# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

try:
    import torch.distributed as dist  # type: ignore
except ImportError:
    dist = None

import transformers
import tokenizers

# Master switch: set LLAVA_SILENT_LOGS=1 to silence all print-based debug output in this file.
_LLAVA_SILENT_LOGS = os.environ.get("LLAVA_SILENT_LOGS", "0") == "1"
if _LLAVA_SILENT_LOGS:
    import builtins as _bt
    def print(*args, **kwargs):  # type: ignore
        return None

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    use_layerwise_vision_fusion: bool = field(default=False)
    vision_router_hidden_dim: Optional[int] = field(default=None)
    vision_router_dropout: float = field(default=0.0)
    vision_fusion_gate_init: float = field(default=1.0)
    router_entropy_loss_enabled: bool = field(default=False)
    router_entropy_loss_coef: float = field(default=0.0)
    train_llm_layers: bool = field(default=False)
    reinit_mm_projector: bool = field(default=False)  # Whether to randomly reinitialize mm_projector
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    fusion_gate_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    extra_save_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Extra step number at which to save a checkpoint in addition to save_steps (e.g. 5195)."}
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def gather_module_state_dict(module: torch.nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
    if module is None:
        return {}

    params = list(module.parameters())
    zero3_enabled = any(hasattr(p, "ds_id") for p in params)
    state_dict: Dict[str, torch.Tensor] = {}

    dist_available = dist is not None and dist.is_available()
    world_rank = dist.get_rank() if dist_available and dist.is_initialized() else 0

    if zero3_enabled and params:
        try:
            from deepspeed import zero
        except ImportError:
            zero3_enabled = False

    if zero3_enabled and params:
        with zero.GatheredParameters(params, modifier_rank=0):
            should_collect = True
            if dist_available and dist.is_initialized():
                should_collect = world_rank == 0
            if should_collect:
                state_dict = {
                    prefix + k: v.detach().cpu()
                    for k, v in module.state_dict().items()
                }
        if dist_available and dist.is_initialized():
            dist.barrier()
            if world_rank != 0:
                state_dict = {}
    else:
        state_dict = {
            prefix + k: v.detach().cpu()
            for k, v in module.state_dict().items()
        }

    return state_dict


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        model = trainer.model.get_model()
        weight_to_save: Dict[str, torch.Tensor] = {}

        weight_to_save.update(
            gather_module_state_dict(getattr(model, "mm_projector", None), "mm_projector.")
        )

        if getattr(model, "layerwise_fusion", None) is not None:
            weight_to_save.update(
                gather_module_state_dict(model.layerwise_fusion, "layerwise_fusion.")
            )

        if getattr(trainer.args, "use_im_start_end", False):
            for key in ['embed_tokens', 'embed_in']:
                module = getattr(model, key, None)
                if module is not None:
                    weight_to_save.update(
                        gather_module_state_dict(module, f"{key}.")
                    )

        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(output_dir)
        else:
            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                else:
                    torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



class FusionGateDebugCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()
        self._last_print_step = -1

    def _is_main_process(self, args):
        return getattr(args, "local_rank", -1) in (-1, 0)

    def _get_layerwise_fusion(self, model):
        if model is None:
            return None
        ds_model = model.module if hasattr(model, "module") else model
        core_model = ds_model.get_model() if hasattr(ds_model, "get_model") else ds_model
        return getattr(core_model, "layerwise_fusion", None)

    def _collect_gamma_values(self, model):
        layerwise_fusion = self._get_layerwise_fusion(model)
        if layerwise_fusion is None or not hasattr(layerwise_fusion, "routers"):
            return None
        gate_values = []
        for idx, router in enumerate(layerwise_fusion.routers):
            if not hasattr(router, "fusion_gate_logit"):
                return None
            gate_logit = maybe_zero_3(
                router.fusion_gate_logit,
                ignore_status=True,
                name=f"layerwise_fusion.routers.{idx}.fusion_gate_logit",
            ).float().view(-1)
            gate_values.append(gate_logit[0])
        gate_logits = torch.stack(gate_values, dim=0)
        return torch.sigmoid(gate_logits)

    def _format_gamma_values(self, gamma_values: torch.Tensor) -> str:
        return "[" + ", ".join(f"{x:.4f}" for x in gamma_values.tolist()) + "]"

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_main_process(args):
            return
        model = kwargs.get("model")
        gamma_values = self._collect_gamma_values(model)
        if gamma_values is None:
            print("[FusionGateDebug] routers.*.fusion_gate_logit not found")
            return
        print("\n[FusionGateDebug] Initial fusion gate gamma (per layer-wise router)")
        print(f"[FusionGateDebug] mean={gamma_values.mean().item():.6f}, min={gamma_values.min().item():.6f}, max={gamma_values.max().item():.6f}")
        print(f"[FusionGateDebug] values={self._format_gamma_values(gamma_values)}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_main_process(args):
            return
        if state.global_step <= 0 or state.global_step == self._last_print_step:
            return
        model = kwargs.get("model")
        gamma_values = self._collect_gamma_values(model)
        if gamma_values is None:
            return
        self._last_print_step = state.global_step
        print(f"\n[FusionGateDebug] step {state.global_step}")
        print(f"[FusionGateDebug] mean={gamma_values.mean().item():.6f}, min={gamma_values.min().item():.6f}, max={gamma_values.max().item():.6f}")
        print(f"[FusionGateDebug] values={self._format_gamma_values(gamma_values)}")


class FusionGateTrainDebugCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()
        self._last_print_step = -1
        self._last_master_value = None

    def _is_main_process(self, args):
        return getattr(args, "local_rank", -1) in (-1, 0)

    def _main_process_only_print(self, args, text):
        if self._is_main_process(args):
            print(text)

    def _collect_fusion_gate_params(self, model):
        if model is None:
            return []
        params = []
        for name, param in model.named_parameters():
            if "layerwise_fusion.routers." in name and name.endswith("fusion_gate_logit"):
                params.append((name, param))
        return sorted(params, key=lambda x: x[0])

    def _gather_master_param_values(self, named_params):
        if not named_params:
            return None
        try:
            from deepspeed.utils import safe_get_full_fp32_param
        except Exception:
            return None
        values = []
        for name, param in named_params:
            try:
                master = safe_get_full_fp32_param(param)
                if master is None:
                    return None
                master = master.detach().float().cpu().view(-1)
                values.append(master[0] if master.numel() > 0 else torch.tensor(float("nan")))
            except Exception as e:
                print(f"[FusionGateTrainDebug] Failed to gather master fusion_gate_logit ({name}): {e}")
                return None
        return torch.stack(values, dim=0)

    def _format_values(self, tensor, precision=4):
        if tensor is None:
            return "None"
        return "[" + ", ".join(f"{float(v):.{precision}f}" for v in tensor.tolist()) + "]"

    def _format_delta_stats(self, tensor, name):
        if tensor is None:
            return f"[FusionGateTrainDebug] {name}: None"
        abs_tensor = tensor.detach().abs().float()
        finite_mask = torch.isfinite(abs_tensor)
        if finite_mask.any():
            valid = abs_tensor[finite_mask]
            max_abs = valid.max().item()
            mean_abs = valid.mean().item()
        else:
            max_abs = float("nan")
            mean_abs = float("nan")
        return (
            f"[FusionGateTrainDebug] {name}: "
            f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        named_params = self._collect_fusion_gate_params(model)
        if not named_params:
            self._main_process_only_print(args, "\n[FusionGateTrainDebug] Parameter check")
            self._main_process_only_print(args, "[FusionGateTrainDebug] routers.*.fusion_gate_logit not found")
            return
        first_name, first_param = named_params[0]
        ds_shape = tuple(getattr(first_param, "ds_shape", tuple(first_param.shape)))
        self._main_process_only_print(args, "\n[FusionGateTrainDebug] Parameter check")
        self._main_process_only_print(
            args,
            f"[FusionGateTrainDebug] {first_name}: requires_grad={first_param.requires_grad}, "
            f"local_shape={tuple(first_param.shape)}, ds_shape={ds_shape}, ds_id={getattr(first_param, 'ds_id', None)}, "
            f"num_gates={len(named_params)}"
        )
        master_value = self._gather_master_param_values(named_params)
        if master_value is not None:
            master_gamma = torch.sigmoid(master_value)
            self._main_process_only_print(args, f"[FusionGateTrainDebug] master_logits ={self._format_values(master_value)}")
            self._main_process_only_print(args, f"[FusionGateTrainDebug] master_gamma  ={self._format_values(master_gamma)}")
            self._last_master_value = master_value.clone()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step == self._last_print_step:
            return
        if state.global_step % max(1, getattr(args, "logging_steps", 1)) != 0:
            return
        self._last_print_step = state.global_step

        model = kwargs.get("model")
        named_params = self._collect_fusion_gate_params(model)
        if not named_params:
            return
        master_value = self._gather_master_param_values(named_params)
        if master_value is None:
            return
        master_delta = master_value - self._last_master_value if master_value is not None and self._last_master_value is not None and self._last_master_value.shape == master_value.shape else None
        master_gamma_delta = torch.sigmoid(master_value) - torch.sigmoid(self._last_master_value) if master_value is not None and self._last_master_value is not None and self._last_master_value.shape == master_value.shape else None
        self._main_process_only_print(args, f"\n[FusionGateTrainDebug] step {state.global_step} parameter snapshot")
        self._main_process_only_print(args, f"[FusionGateTrainDebug] master_logits ={self._format_values(master_value)}")
        self._main_process_only_print(args, f"[FusionGateTrainDebug] master_gamma  ={self._format_values(torch.sigmoid(master_value))}")
        if master_delta is not None:
            self._main_process_only_print(args, self._format_delta_stats(master_delta, "delta_master_logits_stats"))
            self._main_process_only_print(args, f"[FusionGateTrainDebug] delta_master_logits ={self._format_values(master_delta, precision=6)}")
        if master_gamma_delta is not None:
            self._main_process_only_print(args, self._format_delta_stats(master_gamma_delta, "delta_master_gamma_stats"))
            self._main_process_only_print(args, f"[FusionGateTrainDebug] delta_master_gamma  ={self._format_values(master_gamma_delta, precision=6)}")
        self._last_master_value = master_value.clone()


class ClearFusionContextCallback(transformers.TrainerCallback):
    """
    Clear layerwise vision fusion caches before each optimizer.step to avoid reusing stale
    computation graphs across steps (e.g. "Trying to backward through the graph a second time").
    """
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        layerwise = getattr(model, "layerwise_fusion", None)
        if layerwise is not None:
            layerwise.clear_context()


class ExtraSaveStepCallback(transformers.TrainerCallback):
    """Trigger an extra checkpoint save at a given step (e.g. 5195) in addition to periodic save_steps."""
    def on_step_end(self, args, state, control, **kwargs):
        extra = getattr(args, "extra_save_steps", None)
        if extra is not None and state.global_step == extra:
            control.should_save = True
        return control



# class RouterDebugCallback(transformers.TrainerCallback):
#     """
#     Print layer-wise router training state (layerwise_fusion.routers):
#     - At train begin: requires_grad and whether parameters appear in optimizer param groups
#     - Every logging_steps: whether gradients are None and max absolute gradient
#     """

#     def _collect_router_params(self, model):
#         return [(n, p) for n, p in model.named_parameters() if "layerwise_fusion.routers" in n]

#     def on_train_begin(self, args, state, control, **kwargs):
#         model = kwargs.get("model")
#         optimizer = kwargs.get("optimizer")
#         if model is None:
#             return
#         router_params = self._collect_router_params(model)
#         print("\n[RouterDebug] requires_grad check")
#         if not router_params:
#             print("[RouterDebug] No router parameters found (layerwise_fusion.routers)")
#         for n, p in router_params:
#             print(f"{n}: requires_grad={p.requires_grad}")

#         if optimizer is not None:
#             name_by_id = {id(p): n for n, p in model.named_parameters()}
#             print("\n[RouterDebug] Optimizer param groups check")
#             for gi, g in enumerate(optimizer.param_groups):
#                 names = [name_by_id.get(id(p), "<unnamed>") for p in g["params"] if id(p) in name_by_id]
#                 router_names = [n for n in names if n and "layerwise_fusion.routers" in n]
#                 if router_names:
#                     print(f"group {gi} lr={g.get('lr', None)} router_param_count={len(router_names)}")
#                     for n in router_names[:5]:
#                         print("  ", n)
#                 else:
#                     print(f"group {gi} lr={g.get('lr', None)} no router parameters")

#     # Printing at step_end can be cleared by optimizer/zero_grad; backward_end is more reliable
#     def on_backward_end(self, args, state, control, **kwargs):
#         model = kwargs.get("model")
#         if model is None:
#             return
#         # Throttle by logging_steps
#         if state.global_step % max(1, args.logging_steps) != 0:
#             return
#         router_params = self._collect_router_params(model)
#         print(f"\n[RouterDebug] step {state.global_step} gradient check (after backward)")
#         for n, p in router_params[:10]:  # First 10 only to avoid huge logs
#             has_grad = p.grad is not None
#             max_val = p.grad.abs().max().item() if has_grad else None
#             print(f"{n}: grad_none={not has_grad}, max_abs_grad={max_val}")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self._lengths_cache = None
        self._modality_lengths_cache = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        if self._lengths_cache is not None:
            return self._lengths_cache
        length_list = []
        n = len(self.list_data_dict)
        for i, sample in enumerate(self.list_data_dict):
            if (i + 1) % 200000 == 0:
                rank0_print(f"Computing lengths... {i + 1}/{n}")
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        self._lengths_cache = length_list
        return length_list

    @property
    def modality_lengths(self):
        if self._modality_lengths_cache is not None:
            return self._modality_lengths_cache
        length_list = []
        n = len(self.list_data_dict)
        for i, sample in enumerate(self.list_data_dict):
            if (i + 1) % 200000 == 0:
                rank0_print(f"Computing modality_lengths... {i + 1}/{n}")
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        self._modality_lengths_cache = length_list
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # Log CUDA_VISIBLE_DEVICES for multi-GPU sanity checks
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    if training_args.local_rank in [-1, 0]:
        print(f"[GPU Check] CUDA_VISIBLE_DEVICES={cuda_visible}")
        if torch.cuda.is_available():
            print(f"[GPU Check] PyTorch sees {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"[GPU Check] Logical GPU {i}: {torch.cuda.get_device_name(i)}")
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.train_llm_layers = getattr(model_args, "train_llm_layers", False)
        model.config.freeze_vision_tower = getattr(model_args, "freeze_vision_tower", False)
        model.config.router_entropy_loss_enabled = bool(
            getattr(model_args, "router_entropy_loss_enabled", False)
            and getattr(model_args, "use_layerwise_vision_fusion", False)
        )
        model.config.router_entropy_loss_coef = float(getattr(model_args, "router_entropy_loss_coef", 0.0))
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            layerwise_fusion = getattr(model.get_model(), "layerwise_fusion", None)
            if layerwise_fusion is not None:
                for p in layerwise_fusion.parameters():
                    p.requires_grad = True
            if getattr(model_args, "train_llm_layers", False):
                core_model = model.get_model()
                for name, param in core_model.named_parameters():
                    if not param.requires_grad:
                        if name.startswith("vision_tower"):
                            continue
                        if name.startswith("mm_projector") or name.startswith("layerwise_fusion"):
                            continue
                        param.requires_grad = True
        else:
            if getattr(model_args, "freeze_vision_tower", False):
                vision_tower = model.get_model().get_vision_tower()
                if vision_tower is not None:
                    if isinstance(vision_tower, (list, tuple)):
                        towers = vision_tower
                    else:
                        towers = [vision_tower]
                    for tower in towers:
                        tower.requires_grad_(False)
                        for p in tower.parameters():
                            p.requires_grad = False

        # reinit_mm_projector is handled in llava_arch.initialize_vision_modules at model build time
        # to avoid DeepSpeed ZeRO-3 sharding issues if reinit happened later

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    if os.environ.get("LLAVA_DEBUG_FUSION_GATES", "0") == "1":
        trainer.add_callback(FusionGateDebugCallback())
    if os.environ.get("LLAVA_DEBUG_FUSION_GATE_TRAIN", "0") == "1":
        trainer.add_callback(FusionGateTrainDebugCallback())
    trainer.add_callback(ClearFusionContextCallback())
    trainer.add_callback(ExtraSaveStepCallback())

    # ==================== Parameter counts before training ====================
    if training_args.local_rank in [-1, 0]:
        # Under DeepSpeed ZeRO-3, tensors are sharded; use ds_numel for full parameter size
        def _get_numel(p):
            if hasattr(p, 'ds_numel'):  # ZeRO-3 partitioned parameter
                return p.ds_numel
            return p.numel()

        def _count_params(module, trainable_only=False):
            if module is None:
                return 0
            return sum(_get_numel(p) for p in module.parameters() if (p.requires_grad or not trainable_only))

        total_params = sum(_get_numel(p) for p in model.parameters())
        trainable_params = sum(_get_numel(p) for p in model.parameters() if p.requires_grad)

        # mm_projector
        mm_proj = getattr(model.get_model(), "mm_projector", None)
        mm_all = _count_params(mm_proj, False)
        mm_train = _count_params(mm_proj, True)

        # layerwise vision fusion (layer-wise routers, shared cross-attention, etc.)
        lf = getattr(model.get_model(), "layerwise_fusion", None)
        lf_all = _count_params(lf, False)
        lf_train = _count_params(lf, True)
        routers_all = routers_train = 0
        cls_all = cls_train = post_all = post_train = 0
        cross_all = cross_train = 0
        if lf is not None:
            if hasattr(lf, "routers"):
                routers_all = sum(_count_params(r, False) for r in lf.routers)
                routers_train = sum(_count_params(r, True) for r in lf.routers)
            if hasattr(lf, "cls_expert"):
                cls_all = _count_params(lf.cls_expert, False)
                cls_train = _count_params(lf.cls_expert, True)
            elif hasattr(lf, "depth_expert"):
                cls_all = _count_params(lf.depth_expert, False)
                cls_train = _count_params(lf.depth_expert, True)
            if hasattr(lf, "post_fusion_experts"):
                post_all = _count_params(lf.post_fusion_experts, False)
                post_train = _count_params(lf.post_fusion_experts, True)
            if hasattr(lf, "shared_cross_attn"):
                cross_all = _count_params(lf.shared_cross_attn, False)
                cross_train = _count_params(lf.shared_cross_attn, True)

        # vision backbone (ViT)
        vt = model.get_model().get_vision_tower() if hasattr(model.get_model(), "get_vision_tower") else None
        vt_all = _count_params(vt, False) if vt else 0
        vt_train = _count_params(vt, True) if vt else 0

        # LLM backbone (excluding vision_tower, mm_projector, layerwise_fusion)
        llm_all = total_params - vt_all - mm_all - lf_all
        llm_train = trainable_params - vt_train - mm_train - lf_train

        # Detect DeepSpeed ZeRO-3
        is_zero3 = any(hasattr(p, 'ds_numel') for p in model.parameters())

        print("\n" + "=" * 60)
        print("            Pre-training parameter statistics (DGP / layer-wise fusion)")
        if is_zero3:
            print("        (DeepSpeed ZeRO-3: counts use full ds_numel per parameter)")
        print("=" * 60)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}  ({trainable_params/total_params*100:.2f}%)")
        print("-" * 60)
        print(f"vision_tower:         total={vt_all:,}, trainable={vt_train:,}")
        print(f"mm_projector:         total={mm_all:,}, trainable={mm_train:,}")
        print(f"layerwise_fusion:     total={lf_all:,}, trainable={lf_train:,}")
        if lf is not None:
            print(f"     - layer-wise routers (MLP): total={routers_all:,}, trainable={routers_train:,}")
            print(f"     - shared_cross_attention:  total={cross_all:,}, trainable={cross_train:,}")
            print(f"     - ViT CLS / residual path: total={cls_all:,}, trainable={cls_train:,}")
            print(f"     - post_fusion_experts:      total={post_all:,}, trainable={post_train:,}")
        print(f"LLM (remainder):      total={llm_all:,}, trainable={llm_train:,}")
        print("-" * 60)
        print("Trainable fraction by module (of all trainable params):")
        def _ratio(x): return (x / trainable_params * 100) if trainable_params else 0
        print(f"     vision_tower:     {_ratio(vt_train):.4f}%")
        print(f"     mm_projector:     {_ratio(mm_train):.4f}%")
        print(f"     layerwise_fusion: {_ratio(lf_train):.4f}%")
        print(f"     LLM (remainder): {_ratio(llm_train):.4f}%")
        print("-" * 60)
        print("Trainable parameter names:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"     {n}")
        print("=" * 60 + "\n")
    # ==================== End parameter statistics ====================

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
