#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from contextlib import nullcontext

import math
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple

# Global switch: set LLAVA_SILENT_LOGS=1 to silence all print-based debug output in this file.
_LLAVA_SILENT_LOGS = os.environ.get("LLAVA_SILENT_LOGS", "0") == "1"
if _LLAVA_SILENT_LOGS:
    import builtins as _bt
    def print(*args, **kwargs):  # type: ignore
        return None

# Sequence / shared cross-attention / DecoderLayerRouter debug: when LLAVA_DEBUG_SEQ=1, print one sample per batch (sequence, labels, cross-attn, router, etc.); capped at the first N batches to limit log volume.
_LLAVA_DEBUG_SEQ = os.environ.get("LLAVA_DEBUG_SEQ", "0") == "1"
_LLAVA_DEBUG_SEQ_MAX_BATCHES = 2  # Max number of batches to print (per-process debug cap).

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class RMSNorm(nn.Module):
    """RMSNorm matching LlamaRMSNorm in GlimpsePrune; used for glimpse-token LE normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class LayerwiseDepthExpert(nn.Module):
    """Lightweight per-layer depth expert: ViT hidden states pass through LN + Linear into the LLM hidden size."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.proj_ln = nn.LayerNorm(in_dim)
        self.proj_linear = nn.Linear(in_dim, out_dim)
        self._debug_printed = False

    def project(self, tensor: torch.Tensor, ref_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        x = self.proj_ln(tensor)
        projected = self.proj_linear(x)
        if ref_dtype is not None:
            projected = projected.to(ref_dtype)
        else:
            projected = projected.to(tensor.dtype)
        return projected

    def forward(self, hidden_states):
        outputs = []
        total_layers = min(len(hidden_states), self.num_layers)
        for idx in range(total_layers):
            layer_hidden = hidden_states[idx]
            projected = self.project(layer_hidden, ref_dtype=layer_hidden.dtype)

            if not self._debug_printed:
                batch_shape = tuple(layer_hidden.shape)
                sample_shape = tuple(layer_hidden[0].shape) if layer_hidden.ndim >= 2 else batch_shape
                proj_batch_shape = tuple(projected.shape)
                proj_sample_shape = tuple(projected[0].shape) if projected.ndim >= 2 else proj_batch_shape
                print(f"[LayerwiseDepthExpert] ViT layer {idx} raw shape: {batch_shape}, first-sample: {sample_shape}")
                print(f"[LayerwiseDepthExpert] ViT layer {idx} after projector: {proj_batch_shape}, first-sample: {proj_sample_shape}")

            outputs.append(projected)

        if not self._debug_printed:
            self._debug_printed = True
        return outputs


class DecoderLayerRouter(nn.Module):
    """
    Router design A: Q = glimpse-token hidden state at the current DecoderLayerRouter depth; K/V = 24 ViT CLS tokens.
    Uses a **shared** cross-attention module (injected from LayerwiseVisionFusion); this class holds only the **per-layer**
    DecoderLayerRouter head: LN → FFN(D→h→24) → LN → softmax.
    """
    def __init__(self, hidden_dim: int, num_vision_layers: int, router_hidden_dim: Optional[int] = None,
                 dropout: float = 0.0, _num_heads: int = 8, fusion_gate_init: float = 1.0):
        super().__init__()
        self.num_vision_layers = num_vision_layers
        self.hidden_dim = hidden_dim
        # Do not instantiate cross-attn here; LayerwiseVisionFusion passes the shared shared_cross_attn.
        # Per-layer Router MLP (not shared): LN → FFN(D → h → 24); zero-init on the last linear for near-uniform softmax at init.
        router_hidden = router_hidden_dim or min(hidden_dim, 512)
        self.router_ln = nn.LayerNorm(hidden_dim)
        # Explicit init for traceability (LayerNorm defaults: weight=1, bias=0; unchanged).
        torch.nn.init.ones_(self.router_ln.weight)
        torch.nn.init.zeros_(self.router_ln.bias)
        self.router_fc1 = nn.Linear(hidden_dim, router_hidden)
        torch.nn.init.kaiming_uniform_(self.router_fc1.weight, a=5**0.5, mode='fan_in', nonlinearity='leaky_relu')
        if self.router_fc1.bias is not None:
            torch.nn.init.zeros_(self.router_fc1.bias)
        self.router_act = nn.GELU()
        self.router_dropout = nn.Dropout(dropout)
        self.router_fc2 = nn.Linear(router_hidden, num_vision_layers)
        self.router_out_ln = nn.LayerNorm(num_vision_layers)
        torch.nn.init.ones_(self.router_out_ln.weight)
        torch.nn.init.zeros_(self.router_out_ln.bias)
        torch.nn.init.zeros_(self.router_fc2.weight)
        torch.nn.init.zeros_(self.router_fc2.bias)
        # Per-layer fusion gate (sigmoid): gamma_l = sigmoid(fusion_gate_logit).
        gate_prob = min(max(float(fusion_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_prob / (1.0 - gate_prob))
        # ZeRO-3 is unstable for 0-d scalars; use shape (1,) for robust sharding.
        self.fusion_gate_logit = nn.Parameter(torch.full((1,), gate_logit, dtype=torch.float32))

    def keep_fusion_gate_fp32(self):
        self.fusion_gate_logit.data = self.fusion_gate_logit.data.to(dtype=torch.float32)

    def forward(self, glimpse_hidden: torch.Tensor, cls_kv: torch.Tensor,
                cross_attn: nn.MultiheadAttention, layer_idx: int = -1,
                debug_this_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            glimpse_hidden: [B, 1, D] — hidden state at the glimpse-token position (Q).
            cls_kv: [B, 24, D] — projected ViT CLS tokens plus layer-id embeddings (K/V).
            cross_attn: shared nn.MultiheadAttention used by all decoder layers.
            layer_idx: current LLM layer index (debug only).
            debug_this_batch: if True, emit cross-attn / DecoderLayerRouter debug for one sample in this batch.

        Returns:
            weights: [B, num_vision_layers]
            gamma: fusion gate (sigmoid), shape [1].
        """
        # Shared cross-attention: Q=glimpse_hidden, K=V=cls_kv (no Q-side pos enc.; layer_id_embeddings act as 24 "slots" on KV).
        q_in = glimpse_hidden   # [B, 1, D]
        kv_in = cls_kv          # [B, 24, D]
        attn_out, _ = cross_attn(glimpse_hidden, cls_kv, cls_kv, need_weights=True)  # [B, 1, D]
        residual = attn_out + glimpse_hidden  # [B, 1, D]

        if _LLAVA_DEBUG_SEQ and debug_this_batch and q_in.shape[0] > 0:
            q_norm = round(q_in[0].norm().item(), 4)
            attn_norm = round(attn_out[0].norm().item(), 4)
            res_norm = round(residual[0].norm().item(), 4)
            print("[DEBUG_SEQ] Cross-Attn I/O (layer=%d, sample 0): Q_in %s -> attn_out %s -> residual %s" % (
                layer_idx, tuple(q_in[0:1].shape), tuple(attn_out[0:1].shape), tuple(residual[0:1].shape)))
            print("[DEBUG_SEQ] Cross-Attn tensor norms: |Q|=%.4f, |attn_out|=%.4f, |residual|=%.4f" % (q_norm, attn_norm, res_norm))
            print("[DEBUG_SEQ] Cross-Attn K/V input: %s (24 ViT CLS + layer_id_embeddings; no Q-side positional encoding)" % (tuple(kv_in[0:1].shape),))

        # Router MLP: LN → FFN → logits
        x = self.router_ln(residual)
        if _LLAVA_DEBUG_SEQ and debug_this_batch and x.shape[0] > 0:
            print("[DEBUG_SEQ] Router tensor flow (layer=%d, sample 0): residual %s -> LN %s" % (
                layer_idx, tuple(residual[0:1].shape), tuple(x[0:1].shape)))
        x = self.router_fc1(x)
        if _LLAVA_DEBUG_SEQ and debug_this_batch and x.shape[0] > 0:
            print("[DEBUG_SEQ] Router tensor flow (layer=%d, sample 0): Linear1 out %s (D→h)" % (
                layer_idx, tuple(x[0:1].shape)))
        x = self.router_act(x)
        x = self.router_dropout(x)
        logits = self.router_fc2(x)  # [B, 1, 24]
        if _LLAVA_DEBUG_SEQ and debug_this_batch and logits.shape[0] > 0:
            print("[DEBUG_SEQ] Router tensor flow (layer=%d, sample 0): Linear2 out %s (h→24)" % (
                layer_idx, tuple(logits[0:1].shape)))
        logits = self.router_out_ln(logits)
        if _LLAVA_DEBUG_SEQ and debug_this_batch and logits.shape[0] > 0:
            print("[DEBUG_SEQ] Router tensor flow (layer=%d, sample 0): LN_out %s" % (
                layer_idx, tuple(logits[0:1].shape)))
        logits_squeezed = logits.squeeze(1)  # [B, 24]

        logits_float = logits_squeezed.float()
        logits_stable = logits_float - logits_float.max(dim=-1, keepdim=True).values
        weights = torch.softmax(logits_stable, dim=-1).to(glimpse_hidden.dtype)

        if not torch.isfinite(weights).all():
            weights = torch.full_like(weights, 1.0 / weights.shape[-1])

        gamma = torch.sigmoid(self.fusion_gate_logit.float())

        if _LLAVA_DEBUG_SEQ and debug_this_batch and weights.shape[0] > 0:
            w0 = weights[0].detach().float().cpu().tolist()
            w0_fmt = [round(x, 6) for x in w0]
            print("[DEBUG_SEQ] Router internals (layer=%d): residual -> LN->FFN->logits->softmax; sample 0 full 24-d weights: %s" % (layer_idx, w0_fmt))
            print("[DEBUG_SEQ] Router gate (layer=%d): gamma=%.6f" % (layer_idx, float(gamma.detach().float().cpu().item())))
        return weights, gamma


class LayerwiseVisionFusion(nn.Module):
    """
    Design A: one glimpse token (after the question, before the answer) + DecoderLayerRouter with shared cross-attention over 24 ViT CLS tokens.
    Matches GlimpsePrune: before insertion, glimpse embeddings pass le_proj → le_norm → le_dropout.
    """
    def __init__(self, hidden_dim: int, num_vision_layers: int, num_decoder_layers: int,
                 router_hidden_dim: Optional[int] = None, router_dropout: float = 0.0,
                 vision_hidden_dim: Optional[int] = None,
                 le_dropout_prob: float = 0.0, rms_norm_eps: float = 1e-6,
                 fusion_gate_init: float = 1.0):
        super().__init__()
        vision_hidden_dim = vision_hidden_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.vision_hidden_dim = vision_hidden_dim
        # Shared CLS projector (builds K/V for the layer-wise router from ViT CLS tokens).
        self.cls_expert = LayerwiseDepthExpert(vision_hidden_dim, hidden_dim, num_vision_layers)
        # Per-LLM-layer post-fusion expert / projector: fuse in ViT hidden size, then map to LLM width for residual injection.
        self.post_fusion_experts = nn.ModuleList([
            LayerwiseDepthExpert(vision_hidden_dim, hidden_dim, 1)
            for _ in range(num_decoder_layers)
        ])
        # Shared cross-attention (~67M params, one copy for all decoder layers); per layer: DecoderLayerRouter (LN+FFN+LN).
        num_heads = 8
        self.shared_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=router_dropout,
            bias=False,
            batch_first=True,
        )
        self.routers = nn.ModuleList([
            DecoderLayerRouter(
                hidden_dim,
                num_vision_layers,
                router_hidden_dim,
                router_dropout,
                _num_heads=num_heads,
                fusion_gate_init=fusion_gate_init,
            )
            for _ in range(num_decoder_layers)
        ])

        # Single learnable glimpse token embedding (GlimpsePrune: normal mean=0, std=0.02).
        self.glimpse_embedding = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.glimpse_embedding, mean=0.0, std=0.02)
        # Pre-insertion transform: le_proj → le_norm → le_dropout (GlimpsePrune: Xavier on proj, norm weights ones).
        self.le_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.le_proj.weight)
        nn.init.zeros_(self.le_proj.bias)
        self.le_norm = RMSNorm(hidden_dim, eps=rms_norm_eps)  # RMSNorm.weight initialized to ones in RMSNorm.
        self.le_dropout = nn.Dropout(le_dropout_prob)
        # Layer-id embeddings for 24 ViT blocks (same scale as the glimpse token).
        self.layer_id_embeddings = nn.Parameter(torch.empty(num_vision_layers, hidden_dim))
        nn.init.normal_(self.layer_id_embeddings, mean=0.0, std=0.02)
        self._cached_projected_cls_tokens: Optional[List[torch.Tensor]] = None
        self._cached_patch_features: Optional[List[torch.Tensor]] = None
        self._pending_vision_split_sizes: Optional[List[int]] = None
        self._context = None
        self.num_vision_layers = num_vision_layers
        self.num_decoder_layers = num_decoder_layers

        self._collect_router_weights = False
        self._current_sample_weights = {}
        self._all_samples_weights = []
        self._latest_router_weights = {}

    def keep_fusion_gate_fp32(self):
        # Fusion gate gammas are few scalars; keep fp32 so early updates are not washed out under bf16/fp16.
        for router in self.routers:
            router.keep_fusion_gate_fp32()

    def get_processed_glimpse_embedding(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """GlimpsePrune path: le_proj → le_norm → le_dropout; returns [1, hidden_dim]."""
        le = self.glimpse_embedding.to(device=device, dtype=dtype)
        le = self.le_proj(le)
        le = self.le_norm(le)
        le = self.le_dropout(le)
        return le

    def enable_router_weight_collection(self):
        """Enable collection of per-layer router weight vectors."""
        self._collect_router_weights = True
        self._current_sample_weights = {}
        self._all_samples_weights = []
        print("[RouterWeightCollector] Router weight collection enabled.")
    
    def disable_router_weight_collection(self):
        """Disable router weight collection."""
        self._collect_router_weights = False
        print("[RouterWeightCollector] Router weight collection disabled.")
    
    def finalize_current_sample(self):
        """Finalize router weights for the current sample and append to the run list."""
        if self._collect_router_weights and self._current_sample_weights:
            self._all_samples_weights.append(self._current_sample_weights.copy())
            self._current_sample_weights = {}
    
    def get_all_router_weights(self):
        """Return router weights for all collected samples."""
        return self._all_samples_weights

    def reset_latest_router_weights(self):
        """Clear router-weight cache for the current forward (used by the router entropy regularizer)."""
        self._latest_router_weights = {}

    def get_latest_router_weights(self):
        """Per-layer router softmax weights for the current forward (gradients retained)."""
        return self._latest_router_weights
    
    def get_router_weights_statistics(self):
        """
        Aggregate router-weight statistics across collected samples (layer-wise DecoderLayerRouter only; no pre-router).
        Returns:
            {
            'per_sample': [...],  # [num_samples, num_llm_layers, num_vit_layers]
            'mean': np.array, 'std': np.array,
            'num_samples': int, 'num_llm_layers': int, 'num_vit_layers': int
            }
        """
        import numpy as np
        if not self._all_samples_weights:
            return None
        num_samples = len(self._all_samples_weights)
        num_llm_layers = self.num_decoder_layers
        num_vit_layers = self.num_vision_layers
        all_weights = np.zeros((num_samples, num_llm_layers, num_vit_layers))
        for sample_idx, sample_weights in enumerate(self._all_samples_weights):
            for layer_key, weights in sample_weights.items():
                if isinstance(layer_key, int):
                    if isinstance(weights, torch.Tensor):
                        w = weights[0].detach().cpu().numpy()
                    else:
                        w = np.array(weights)
                    all_weights[sample_idx, layer_key, :] = w
        return {
            'per_sample': all_weights,
            'mean': np.mean(all_weights, axis=0),
            'std': np.std(all_weights, axis=0),
            'num_samples': num_samples,
            'num_llm_layers': num_llm_layers,
            'num_vit_layers': num_vit_layers,
        }

    def encode(self, patch_features: List[torch.Tensor], cls_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """Cache ViT features: patch tokens for layerwise fusion in ViT width; CLS projected to LLM width for router K/V."""
        projected_cls_tokens = [
            self.cls_expert.project(cls_token, ref_dtype=patch_features[0].dtype)
            for cls_token in cls_tokens
        ]
        if _LLAVA_DEBUG_SEQ and len(patch_features) > 0 and len(projected_cls_tokens) > 0:
            p0 = patch_features[0]
            c0 = cls_tokens[0]
            pc0 = projected_cls_tokens[0]
            print(
                "[DEBUG_SEQ] encode cache: patch_features[0]=%s, cls_tokens[0]=%s -> projected_cls_tokens[0]=%s (cls_expert, ViT dim -> LLM dim)"
                % (tuple(p0.shape), tuple(c0.shape), tuple(pc0.shape))
            )
        self._cached_projected_cls_tokens = projected_cls_tokens
        self._cached_patch_features = patch_features
        return patch_features

    def set_vision_split_sizes(self, split_sizes: Optional[List[int]]):
        """Record how many encoded images belong to each LLM sample (for multi-image)."""
        self._pending_vision_split_sizes = list(split_sizes) if split_sizes is not None else None

    def set_context(self, image_token_indices: List[torch.Tensor], token_stats: Optional[List[Dict[str, int]]] = None):
        if self._cached_patch_features is None or len(self._cached_patch_features) == 0:
            self._context = None
            return
        counts = torch.tensor(
            [idx.numel() for idx in image_token_indices],
            device=self._cached_patch_features[0].device if self._cached_patch_features else "cpu",
            dtype=torch.long
        )
        if counts.sum() == 0:
            self._context = {'has_tokens': False}
            return
        vision_b = int(self._cached_patch_features[0].shape[0])
        llm_b = len(image_token_indices)
        split_sizes = self._pending_vision_split_sizes
        if split_sizes is None:
            if vision_b == llm_b:
                split_sizes = [1] * llm_b
            elif llm_b > 0 and vision_b % llm_b == 0:
                split_sizes = [vision_b // llm_b] * llm_b
            else:
                split_sizes = None
        if split_sizes is not None and sum(split_sizes) != vision_b:
            split_sizes = None
        self._context = {
            'patch_features': [token.clone() for token in (self._cached_patch_features or [])],
            'projected_cls_tokens': [token.clone() for token in (self._cached_projected_cls_tokens or [])],
            'indices': [idx.clone() for idx in image_token_indices],
            'counts': counts,
            'token_stats': token_stats or [],
            'vision_split_sizes': split_sizes,
            'has_tokens': True
        }
        self._pending_vision_split_sizes = None

    def clear_context(self):
        self._context = None
        self._cached_projected_cls_tokens = None
        self._cached_patch_features = None
        # Keep _pending_vision_split_sizes: it is set right before encode_images(),
        # while training encode() may call clear_context() first.

    @property
    def has_context(self) -> bool:
        return self._context is not None and self._context.get('has_tokens', False)

    def _compute_fused(self, hidden_states: torch.Tensor, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Q = glimpse-token hidden state at this depth; K/V = 24 ViT CLS tokens plus layer-id embeddings; shared cross-attn → routing weights → layerwise patch fusion.

        Supports multi-image samples where vision batch (total images) != LLM batch.
        Returns fused with shape [llm_B, N_img_tokens, D] (tokens from all images of each sample concatenated).
        """
        if not self.has_context:
            return None
        projected_cls_tokens = self._context.get('projected_cls_tokens')
        if projected_cls_tokens is None or len(projected_cls_tokens) == 0:
            return None

        batch_size, seq_len, _ = hidden_states.shape
        token_stats_ctx = self._context.get('token_stats', [])
        glimpse_pos_list = [s.get('glimpse_pos', 0) for s in token_stats_ctx] if token_stats_ctx else []
        if len(glimpse_pos_list) != batch_size:
            glimpse_pos_list = [0] * batch_size
        glimpse_pos_list = [min(p, seq_len - 1) for p in glimpse_pos_list]

        # 24 projected ViT CLS tokens + layer-id embeddings -> [N_img, 24, D]
        vision_summary = []
        for projected_cls in projected_cls_tokens:
            vision_summary.append(projected_cls.to(hidden_states.dtype).squeeze(1))
        cls_proj = torch.stack(vision_summary, dim=1)  # [N_img, 24, D]
        layer_emb = self.layer_id_embeddings.to(hidden_states.dtype).unsqueeze(0)
        cls_kv = cls_proj + layer_emb
        vision_b = cls_kv.shape[0]

        # Glimpse-token hidden states per LLM sample -> [llm_B, 1, D]
        device = hidden_states.device
        indices = torch.tensor(glimpse_pos_list, device=device, dtype=torch.long)
        glimpse_hidden = hidden_states[torch.arange(batch_size, device=device), indices, :].unsqueeze(1)

        split_sizes = self._context.get('vision_split_sizes')
        if vision_b != batch_size:
            if split_sizes is None or sum(split_sizes) != vision_b or len(split_sizes) != batch_size:
                # Cannot safely align multi-image features with LLM batch; skip fusion.
                return None
            glimpse_chunks = []
            for b, n_img in enumerate(split_sizes):
                if n_img <= 0:
                    continue
                glimpse_chunks.append(glimpse_hidden[b:b + 1].expand(n_img, -1, -1))
            glimpse_for_router = torch.cat(glimpse_chunks, dim=0)  # [N_img, 1, D]
        else:
            glimpse_for_router = glimpse_hidden
            if split_sizes is None:
                split_sizes = [1] * batch_size

        debug_this_batch = getattr(self, "_debug_this_batch", False)
        weights, gamma = self.routers[layer_idx](
            glimpse_for_router,
            cls_kv,
            self.shared_cross_attn,
            layer_idx=layer_idx,
            debug_this_batch=debug_this_batch,
        )
        self._latest_router_weights[layer_idx] = weights
        if self._collect_router_weights:
            # Keep one weight vector per LLM sample for logging (mean over images).
            if vision_b != batch_size:
                w_per_sample = []
                offset = 0
                for n_img in split_sizes:
                    if n_img <= 0:
                        w_per_sample.append(weights.new_zeros(weights.shape[-1]))
                        continue
                    w_per_sample.append(weights[offset:offset + n_img].mean(dim=0))
                    offset += n_img
                self._current_sample_weights[layer_idx] = torch.stack(w_per_sample, dim=0).detach().clone()
            else:
                self._current_sample_weights[layer_idx] = weights.detach().clone()
        # Weighted fusion over 24 ViT layers in ViT hidden space -> [N_img, Nv, vision_hidden_dim]
        stacked_raw = torch.stack(self._context['patch_features'], dim=1)  # [N_img, 24, Nv, 1024]
        fused_1024 = torch.sum(weights[:, :, None, None] * stacked_raw, dim=1)
        # Per-LLM-layer post-fusion expert / projector maps fused patches to LLM width for residual injection
        fused_imgs = self.post_fusion_experts[layer_idx].project(fused_1024, ref_dtype=hidden_states.dtype)  # [N_img, Nv, D]
        if not torch.isfinite(fused_imgs).all():
            fused_imgs = torch.zeros_like(fused_imgs)
        if not torch.isfinite(weights).all():
            weights = torch.full_like(weights, 1.0 / weights.shape[-1])

        # Pack image tokens per LLM sample: [llm_B, sum_i Nv_i, D]
        fused_chunks = torch.split(fused_imgs, split_sizes, dim=0)
        fused_list = []
        max_tokens = 0
        for chunk in fused_chunks:
            if chunk.numel() == 0:
                flat = fused_imgs.new_zeros((0, fused_imgs.shape[-1]))
            else:
                flat = chunk.reshape(-1, chunk.shape[-1])
            fused_list.append(flat)
            max_tokens = max(max_tokens, flat.shape[0])
        fused = fused_imgs.new_zeros((batch_size, max_tokens, fused_imgs.shape[-1]))
        for b, flat in enumerate(fused_list):
            if flat.shape[0] > 0:
                fused[b, :flat.shape[0]] = flat

        if _LLAVA_DEBUG_SEQ and debug_this_batch and fused.shape[0] > 0:
            w0 = weights[0].detach().float().cpu()
            topk = min(3, w0.numel())
            top_vals, top_idx = torch.topk(w0, k=topk)
            print("[DEBUG_SEQ] Layer=%d routing: weights shape=%s, top-%d ViT layers=%s, top-%d values=%s" % (
                layer_idx, tuple(weights.shape), topk, top_idx.tolist(), topk, [round(v.item(), 6) for v in top_vals]
            ))
            print("[DEBUG_SEQ] Layer=%d fusion shapes: stacked_raw=%s -> fused_vision=%s -> fused_packed=%s" % (
                layer_idx, tuple(stacked_raw.shape), tuple(fused_1024.shape), tuple(fused.shape)
            ))
        return fused, gamma

    def inject(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor: 
        fused_and_gamma = self._compute_fused(hidden_states, layer_idx)
        if fused_and_gamma is None:
            return hidden_states
        fused, gamma = fused_and_gamma

        updated_hidden = hidden_states.clone()
        gamma = gamma.view(1, 1, 1)
        fused = (gamma * fused.float()).to(hidden_states.dtype)
        for batch_idx, positions in enumerate(self._context['indices']):  # Image-token indices per batch item.
            if positions.numel() == 0:  # No vision tokens in this batch row.
                continue
            count = min(positions.numel(), fused.shape[1])  # Number of image tokens to update for this row.
            if count == 0:
                continue
            debug_this_batch = getattr(self, "_debug_this_batch", False)
            if _LLAVA_DEBUG_SEQ and debug_this_batch and batch_idx == 0:
                before = updated_hidden[batch_idx, positions[:count]].detach()
                before_norms = [round(before[i].norm().item(), 4) for i in range(min(3, before.shape[0]))]
            updated_hidden[batch_idx, positions[:count]] = (
                updated_hidden[batch_idx, positions[:count]] + fused[batch_idx, :count]
            )
            if _LLAVA_DEBUG_SEQ and debug_this_batch and batch_idx == 0:
                after = updated_hidden[batch_idx, positions[:count]].detach()
                after_norms = [round(after[i].norm().item(), 4) for i in range(min(3, after.shape[0]))]
                print("[DEBUG_SEQ] Residual injection (layer=%d, fusion gate gamma=%.6f): first 3 vision-token L2 norms %s -> %s" % (
                    layer_idx, float(gamma.detach().float().cpu().item()), before_norms, after_norms))

        return updated_hidden

    
    def build_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if not self.has_context:
                return
            hidden_states = args[0]
            updated_hidden = self.inject(hidden_states, layer_idx)
            if updated_hidden is hidden_states:
                return
            new_args = (updated_hidden,) + args[1:]
            return new_args, kwargs
        return hook

    
    
    # def build_hook(self, layer_idx: int):  # Optional: forward_pre_hook to inject layerwise-fused vision into LLM hidden states at layer ell.
    #     def hook(module, args, kwargs):
    #         if not self.has_context:
    #             return
    #         hidden_states = args[0]
    #         updated_hidden = self.inject(hidden_states, layer_idx)
    #         if updated_hidden is hidden_states:
    #             return
    #         print(f"[LayerwiseVisionFusion][hook] layer {layer_idx} has_context={self.has_context} grad_enabled={torch.is_grad_enabled()}")
    #         print(f"[LayerwiseVisionFusion][hook] layer {layer_idx} hidden_states requires_grad={hidden_states.requires_grad}")
    #         print(f"[LayerwiseVisionFusion][hook] layer {layer_idx} updated_hidden requires_grad={updated_hidden.requires_grad}")
    #         print(f"[LayerwiseVisionFusion][hook] layer {layer_idx} torch.is_grad_enabled={torch.is_grad_enabled()}")
    #         new_args = (updated_hidden,) + args[1:]
    #         return new_args, kwargs

    #     return hook


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        self.layerwise_fusion: Optional[LayerwiseVisionFusion] = None  # LayerwiseVisionFusion module.
        self._layerwise_hooks_registered = False  # Whether forward hooks are installed on decoder layers.
        self._layerwise_hook_handles = []

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):  # Return the vision backbone (ViT / CLIP).
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def _register_layerwise_hooks(self):  # Register LayerwiseVisionFusion forward_pre_hooks on each decoder layer.
        if self._layerwise_hooks_registered:
            return
        if self.layerwise_fusion is None:
            return
        if not hasattr(self, 'layers'):
            return

        self._layerwise_hook_handles = []
        for layer_idx, layer in enumerate(self.layers):
            handle = layer.register_forward_pre_hook(
                self.layerwise_fusion.build_hook(layer_idx),
                with_kwargs=True
            )
            self._layerwise_hook_handles.append(handle)
        self._layerwise_hooks_registered = True

    def initialize_vision_modules(self, model_args, fsdp=None):  # Build or reload vision tower, mm_projector, optional LayerwiseVisionFusion.
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # Optionally rebuild mm_projector from scratch (random init).
        reinit_mm_projector = getattr(model_args, 'reinit_mm_projector', False)
        if isinstance(reinit_mm_projector, str):
            reinit_mm_projector = reinit_mm_projector.lower() in ('true', '1', 'yes')
        
        if getattr(self, 'mm_projector', None) is None or reinit_mm_projector:
            if reinit_mm_projector and getattr(self, 'mm_projector', None) is not None:
                print("=" * 80)
                print("[reinit_mm_projector] Recreating mm_projector (random init).")
                print("=" * 80)
            
            # New mm_projector (default Linear Kaiming init).
            self.mm_projector = build_vision_projector(self.config)
            
            if reinit_mm_projector:
                # Log fresh parameter stats
                for name, module in self.mm_projector.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"  [new] {name}: shape={tuple(module.weight.shape)}, "
                              f"mean={module.weight.mean().item():.6f}, std={module.weight.std().item():.6f}")
                print("=" * 80)
                print("[reinit_mm_projector] mm_projector will train from scratch to align multi-layer ViT features.")
                print("=" * 80)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        enable_layerwise_fusion = getattr(model_args, 'use_layerwise_vision_fusion', False)
        if enable_layerwise_fusion:
            vision_hidden_layers = getattr(vision_tower.config, 'num_hidden_layers', len(self.mm_projector))
            router_hidden_dim = getattr(model_args, 'vision_router_hidden_dim', None)
            router_dropout = getattr(model_args, 'vision_router_dropout', 0.0)
            le_dropout_prob = getattr(model_args, 'glimpse_le_dropout_prob', 0.0)
            fusion_gate_init = getattr(model_args, 'vision_fusion_gate_init', 1.0)
            rms_norm_eps = getattr(self.config, 'rms_norm_eps', 1e-6)
            if self.layerwise_fusion is None:
                self.layerwise_fusion = LayerwiseVisionFusion(
                    hidden_dim=self.config.hidden_size,
                    num_vision_layers=vision_hidden_layers,
                    num_decoder_layers=self.config.num_hidden_layers,
                    router_hidden_dim=router_hidden_dim,
                    router_dropout=router_dropout,
                    vision_hidden_dim=self.config.mm_hidden_size,
                    le_dropout_prob=le_dropout_prob,
                    rms_norm_eps=rms_norm_eps,
                    fusion_gate_init=fusion_gate_init,
                )
            mm_proj_weight = getattr(self.mm_projector, "weight", None)
            if mm_proj_weight is not None:
                self.layerwise_fusion.to(
                    device=mm_proj_weight.device,
                    dtype=mm_proj_weight.dtype
                )
                self.layerwise_fusion.keep_fusion_gate_fp32()
            self.config.use_layerwise_vision_fusion = True
            self.config.vision_fusion_gate_init = fusion_gate_init
            self._register_layerwise_hooks()
        else:
            self.config.use_layerwise_vision_fusion = False

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            def _load_module_state(module: nn.Module,
                                   state_dict: Dict[str, torch.Tensor],
                                   strict: bool = True):
                if module is None or len(state_dict) == 0:
                    return None

                params = list(module.parameters())
                target_dtype = params[0].dtype if params else None
                if target_dtype is not None:
                    state_dict = {
                        key: value.to(target_dtype)
                        for key, value in state_dict.items()
                    }

                zero3_enabled = any(
                    hasattr(param, "ds_id") for param in params
                )
                context = nullcontext()
                if zero3_enabled:
                    try:
                        from deepspeed import zero
                        context = zero.GatheredParameters(params, modifier_rank=0)
                    except ImportError:
                        zero3_enabled = False
                        context = nullcontext()

                load_result = None
                with context:
                    should_load = True
                    if zero3_enabled and dist is not None and dist.is_initialized():
                        should_load = dist.get_rank() == 0
                    if should_load:
                        load_result = module.load_state_dict(state_dict, strict=strict)

                if zero3_enabled and dist is not None and dist.is_initialized():
                    dist.barrier()

                return load_result

            projector_state = get_w(mm_projector_weights, 'mm_projector')
            _load_module_state(self.mm_projector, projector_state, strict=True)

            if getattr(self.config, 'use_layerwise_vision_fusion', False) and self.layerwise_fusion is not None:
                fusion_state = get_w(mm_projector_weights, 'layerwise_fusion')
                load_info = _load_module_state(self.layerwise_fusion, fusion_state, strict=False)
                if load_info is not None:
                    missing, unexpected = load_info
                    if missing or unexpected:
                        print(
                            "[LayerwiseVisionFusion] load_state_dict: "
                            f"missing={missing}, unexpected={unexpected}"
                        )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        model = self.get_model()
        vision_tower = model.get_vision_tower()

        if images is None:
            if getattr(model, "layerwise_fusion", None) is not None:
                model.layerwise_fusion.clear_context()
            return None

        if getattr(model.config, "use_layerwise_vision_fusion", False) and getattr(model, "layerwise_fusion", None) is not None:
            if model.training:
                model.layerwise_fusion.clear_context()
            with torch.no_grad():
                patch_features, cls_tokens = vision_tower.get_all_layer_features(images)
            patch_features = [x.detach() for x in patch_features]
            cls_tokens = [x.detach() for x in cls_tokens]
            model.layerwise_fusion.encode(patch_features, cls_tokens)
            # LLM layer-0 inputs: penultimate ViT block (-2) after mm_projector
            vit_second_last = patch_features[-2] if len(patch_features) >= 2 else patch_features[-1]
            if _LLAVA_DEBUG_SEQ and getattr(model, "_debug_seq_printed_batches", 0) <= _LLAVA_DEBUG_SEQ_MAX_BATCHES:
                print("[DEBUG_SEQ] Layer-0 input: penultimate ViT block (-2) after mm_projector. vit_second_last.shape=%s" % (tuple(vit_second_last.shape),))
            return model.mm_projector(vit_second_last)

        image_features = vision_tower(images)
        image_features = model.mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        layerwise_fusion = None
        if getattr(self.get_model().config, "use_layerwise_vision_fusion", False):
            layerwise_fusion = getattr(self.get_model(), "layerwise_fusion", None)

        image_token_masks = []
        token_stats_per_sample: List[Dict[str, int]] = []

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            split_sizes = [image.shape[0] for image in images]
            if layerwise_fusion is not None:
                layerwise_fusion.set_vision_split_sizes(split_sizes)
            image_features = self.encode_images(concat_images)
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if layerwise_fusion is not None:
                # Single tensor path: one image per LLM sample.
                layerwise_fusion.set_vision_split_sizes([1] * int(images.shape[0]))
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # ---------- Debug: raw input_ids / lengths / tokens before inserting image and glimpse tokens ----------
        if _LLAVA_DEBUG_SEQ and layerwise_fusion is not None and len(input_ids) > 0:
            model_for_debug = self.get_model()
            debug_batches = getattr(model_for_debug, "_debug_seq_printed_batches", 0)
            if debug_batches < _LLAVA_DEBUG_SEQ_MAX_BATCHES:
                idx0 = 0
                cur_ids = input_ids[idx0]
                cur_labels = labels[idx0]
                total_len = int(cur_ids.shape[0])
                num_img = int((cur_ids == IMAGE_TOKEN_INDEX).sum().item())
                answer_len = int((cur_labels != IGNORE_INDEX).sum().item())
                question_len = total_len - num_img - answer_len
                print("[DEBUG_SEQ] Before image/glimpse insertion: raw input_ids stats (sample 0)")
                print("   total_len=%d, num_image_tokens=%d, question_len=%d, answer_len=%d" % (
                    total_len, num_img, question_len, answer_len))
                print("   input_ids (full): %s" % (cur_ids.tolist(),))
                # Optional: tokenizer subword pieces
                tokenizer = getattr(model_for_debug, "tokenizer", None)
                if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
                    try:
                        tok_list = tokenizer.convert_ids_to_tokens(cur_ids.tolist())
                        print("   tokens (full): %s" % (tok_list,))
                    except Exception as e:
                        print("   tokens (conversion failed): %s" % (str(e),))

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_image_features = cur_image_features.to(cur_input_embeds_1.dtype)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                cur_new_labels = labels[batch_idx]
                if layerwise_fusion is not None:
                    # Align with the image branch: insert G before the first answer token (G precedes A).
                    non_ignore = (cur_new_labels != IGNORE_INDEX)
                    if non_ignore.any():
                        first_answer_pos = non_ignore.nonzero(as_tuple=True)[0][0].item()
                    else:
                        first_answer_pos = cur_new_labels.shape[0]
                    glimpse_emb = layerwise_fusion.get_processed_glimpse_embedding(cur_input_embeds.device, cur_input_embeds.dtype)
                    cur_input_embeds = torch.cat([
                        cur_input_embeds[:first_answer_pos],
                        glimpse_emb,
                        cur_input_embeds[first_answer_pos:],
                    ], dim=0)
                    cur_new_labels = torch.cat([
                        cur_new_labels[:first_answer_pos],
                        torch.full((1,), IGNORE_INDEX, device=cur_new_labels.device, dtype=cur_new_labels.dtype),
                        cur_new_labels[first_answer_pos:],
                    ], dim=0)
                    glimpse_pos = first_answer_pos
                else:
                    glimpse_pos = None
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_new_labels)
                image_token_masks.append(torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device))
                total_tokens = int(cur_input_embeds.shape[0])
                answer_tokens = int((cur_new_labels != IGNORE_INDEX).sum().item())
                stats = {
                    'total_tokens': total_tokens,
                    'vision_tokens': 0,
                    'text_tokens': total_tokens,
                    'question_tokens': total_tokens - answer_tokens,
                    'answer_tokens': answer_tokens,
                }
                if glimpse_pos is not None:
                    stats['glimpse_pos'] = glimpse_pos
                token_stats_per_sample.append(stats)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_image_mask_segments = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_image_mask_segments.append(torch.zeros(
                    cur_input_embeds_no_im[i].shape[0],
                    dtype=torch.bool,
                    device=cur_input_embeds_no_im[i].device
                ))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_features = cur_image_features.to(cur_input_embeds_no_im[i].dtype)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_image_mask_segments.append(torch.ones(
                        cur_image_features.shape[0],
                        dtype=torch.bool,
                        device=cur_image_features.device
                    ))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_image_mask = torch.cat([segment.to(cur_new_input_embeds.device) for segment in cur_new_image_mask_segments], dim=0)

            # Design A: one glimpse token after Q and before A (training: first answer position; generation: end of sequence in prior logic).
            if layerwise_fusion is not None:
                non_ignore = (cur_new_labels != IGNORE_INDEX)
                if non_ignore.any():
                    first_answer_pos = non_ignore.nonzero(as_tuple=True)[0][0].item()
                else:
                    first_answer_pos = cur_new_labels.shape[0]
                glimpse_emb = layerwise_fusion.get_processed_glimpse_embedding(cur_new_input_embeds.device, cur_new_input_embeds.dtype)
                cur_new_input_embeds = torch.cat([
                    cur_new_input_embeds[:first_answer_pos],
                    glimpse_emb,
                    cur_new_input_embeds[first_answer_pos:],
                ], dim=0)
                cur_new_labels = torch.cat([
                    cur_new_labels[:first_answer_pos],
                    torch.full((1,), IGNORE_INDEX, device=cur_new_labels.device, dtype=cur_new_labels.dtype),
                    cur_new_labels[first_answer_pos:],
                ], dim=0)
                cur_new_image_mask = torch.cat([
                    cur_new_image_mask[:first_answer_pos],
                    torch.zeros(1, dtype=torch.bool, device=cur_new_image_mask.device),
                    cur_new_image_mask[first_answer_pos:],
                ], dim=0)
                glimpse_pos = first_answer_pos
            else:
                glimpse_pos = None

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_token_masks.append(cur_new_image_mask)
            total_tokens = int(cur_new_input_embeds.shape[0])
            vision_tokens = int(cur_new_image_mask.sum().item())
            answer_tokens = int((cur_new_labels != IGNORE_INDEX).sum().item())
            question_tokens = total_tokens - vision_tokens - answer_tokens
            stats = {
                'total_tokens': total_tokens,
                'vision_tokens': vision_tokens,
                'text_tokens': total_tokens - vision_tokens,
                'question_tokens': question_tokens,
                'answer_tokens': answer_tokens,
            }
            if glimpse_pos is not None:
                stats['glimpse_pos'] = glimpse_pos
            token_stats_per_sample.append(stats)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            image_token_masks = [mask[:tokenizer_model_max_length] for mask in image_token_masks]
            for idx in range(len(token_stats_per_sample)):
                total_tokens = int(new_input_embeds[idx].shape[0])
                vision_tokens = int(image_token_masks[idx].sum().item())
                answer_tokens = int((new_labels[idx] != IGNORE_INDEX).sum().item())
                question_tokens = total_tokens - vision_tokens - answer_tokens
                token_stats_per_sample[idx]['total_tokens'] = total_tokens
                token_stats_per_sample[idx]['vision_tokens'] = vision_tokens
                token_stats_per_sample[idx]['text_tokens'] = total_tokens - vision_tokens
                token_stats_per_sample[idx]['question_tokens'] = question_tokens
                token_stats_per_sample[idx]['answer_tokens'] = answer_tokens
                if 'glimpse_pos' in token_stats_per_sample[idx]:
                    token_stats_per_sample[idx]['glimpse_pos'] = min(token_stats_per_sample[idx]['glimpse_pos'], total_tokens - 1)

        # ---------- Debug: one sample per batch — token layout / position_ids / labels (capped at N batches) ----------
        if _LLAVA_DEBUG_SEQ and layerwise_fusion is not None and len(new_input_embeds) > 0:
            model_for_debug = self.get_model()
            debug_batches = getattr(model_for_debug, "_debug_seq_printed_batches", 0)
            if debug_batches < _LLAVA_DEBUG_SEQ_MAX_BATCHES:
                idx0 = 0
                cur_len = int(new_input_embeds[idx0].shape[0])
                im_mask = image_token_masks[idx0][:cur_len]
                lab = new_labels[idx0][:cur_len]
                stats0 = token_stats_per_sample[idx0] if idx0 < len(token_stats_per_sample) else {}
                glimpse_pos = stats0.get("glimpse_pos", -1)
                token_types = []
                for i in range(cur_len):
                    if im_mask[i].item():
                        token_types.append("V")
                    elif i == glimpse_pos:
                        token_types.append("G")
                    elif lab[i].item() != IGNORE_INDEX:
                        token_types.append("A")
                    else:
                        token_types.append("Q")
                pos_ids = list(range(cur_len))
                labels_list = lab.tolist()
                model_for_debug._debug_seq_info = {
                    "token_types": token_types,
                    "position_ids": pos_ids,
                    "labels": labels_list,
                    "glimpse_pos": glimpse_pos,
                    "cur_len": cur_len,
                }
                model_for_debug._debug_seq_printed_batches = debug_batches + 1
                layerwise_fusion._debug_this_batch = True
                # Sequence legend; G must use IGNORE_INDEX in labels for the glimpse token.
                sep = ""
                seq_str = sep.join(token_types)
                glimpse_ok = (glimpse_pos >= 0 and glimpse_pos < len(labels_list) and labels_list[glimpse_pos] == IGNORE_INDEX) if glimpse_pos >= 0 else True
                print("[DEBUG_SEQ] Sample 0 layout (V=vision S=system Q=query A=answer G=glimpse P=padding):")
                print(f"   token_types (len={len(token_types)}): {seq_str}")
                print(f"   position_ids (full): {pos_ids}")
                print(f"   labels (full, IGNORE=%d): %s" % (IGNORE_INDEX, str(labels_list)))
                print(f"   glimpse_pos=%s, label at glimpse is IGNORE_INDEX: %s" % (glimpse_pos, glimpse_ok))
                print(f"   Learned glimpse_embedding shape: %s" % (tuple(layerwise_fusion.glimpse_embedding.shape),))
                print(f"   Glimpse token count: 1, position: %s" % (glimpse_pos,))
            else:
                layerwise_fusion._debug_this_batch = False
        if _LLAVA_DEBUG_SEQ and layerwise_fusion is not None and not getattr(layerwise_fusion, "_debug_this_batch", False):
            layerwise_fusion._debug_this_batch = False

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        image_token_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels_padded.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    mask_slice = image_token_masks[i][:cur_len].to(image_token_mask_padded.device)
                    image_token_mask_padded[i, -cur_len:] = mask_slice
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    mask_slice = image_token_masks[i][:cur_len].to(image_token_mask_padded.device)
                    image_token_mask_padded[i, :cur_len] = mask_slice

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        image_token_indices = [
            torch.nonzero(image_token_mask_padded[i], as_tuple=False).view(-1)
            for i in range(batch_size)
        ]
        if layerwise_fusion is not None:
            layerwise_fusion.set_context(image_token_indices, token_stats_per_sample)

        # ---------- Debug: after padding (P / position_ids / labels / attention_mask) ----------
        if _LLAVA_DEBUG_SEQ and layerwise_fusion is not None:
            model_for_debug = self.get_model()
            if getattr(layerwise_fusion, "_debug_this_batch", False):
                idx0 = 0
                max_len_dbg = int(new_input_embeds.shape[1])
                attn0 = attention_mask[idx0].tolist()
                pos0 = position_ids[idx0].tolist()
                lab0 = new_labels_padded[idx0].tolist() if _labels is not None else []
                # Append P from unpadded token_types to show right-padding
                token_types = model_for_debug._debug_seq_info.get("token_types", []) if hasattr(model_for_debug, "_debug_seq_info") else []
                cur_len = len(token_types)
                token_types_padded = token_types + ["P"] * max(0, max_len_dbg - cur_len)
                if len(token_types_padded) > max_len_dbg:
                    token_types_padded = token_types_padded[:max_len_dbg]
                print("[DEBUG_SEQ] After padding (V=vision S=system Q=query A=answer G=glimpse P=padding; S and Q both tagged Q here):")
                print("   token_types_padded (full): %s" % ("".join(token_types_padded),))
                print("   position_ids (full): %s" % (str(pos0),))
                if _labels is not None:
                    print("   labels (full, IGNORE=%d): %s" % (IGNORE_INDEX, str(lab0)))
                print("   attention_mask (full): %s" % (str(attn0),))

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
