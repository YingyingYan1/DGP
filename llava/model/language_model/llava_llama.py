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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._last_loss_breakdown = {}

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _compute_router_entropy_regularizer(self, layerwise_fusion_module) -> Optional[torch.Tensor]:
        router_weights = layerwise_fusion_module.get_latest_router_weights()
        if not router_weights:
            return None

        entropy_terms = []
        eps = 1e-8
        for layer_idx in sorted(router_weights.keys()):
            weights = router_weights[layer_idx]
            if weights is None or weights.numel() == 0:
                continue
            mean_weights = weights.float().mean(dim=0)
            mean_weights = mean_weights / mean_weights.sum().clamp_min(eps)
            entropy_terms.append(torch.sum(mean_weights * torch.log(mean_weights.clamp_min(eps))))

        if not entropy_terms:
            return None
        return torch.stack(entropy_terms).mean()

    # Prefer non-reentrant gradient checkpointing so LayerwiseVisionFusion hooks stay on the autograd graph.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        # Force non-reentrant so recomputation preserves gradients through hooks.
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
        return super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # Whether to return hidden states from the stack.
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        layerwise_fusion_module = getattr(self.model, "layerwise_fusion", None) if getattr(self.model.config, "use_layerwise_vision_fusion", False) else None
        if layerwise_fusion_module is not None:
            layerwise_fusion_module.reset_latest_router_weights()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # Pure-text step (images=None and fresh inputs_embeds): drop stale vision fusion context from the prior batch.
        # If inputs_embeds come from generate() (image already baked in), skip clearing — use _inputs_embeds_from_generate.
        _inputs_embeds_from_generate = getattr(self, '_inputs_embeds_from_generate', False)
        if layerwise_fusion_module is not None and images is None and not _inputs_embeds_from_generate:
            layerwise_fusion_module.clear_context()

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_loss = outputs.loss
        router_entropy_reg = None
        router_entropy_loss_enabled = bool(getattr(self.config, "router_entropy_loss_enabled", False))
        router_entropy_loss_coef = float(getattr(self.config, "router_entropy_loss_coef", 0.0))
        if (
            self.training
            and layerwise_fusion_module is not None
            and router_entropy_loss_enabled
            and router_entropy_loss_coef != 0.0
        ):
            router_entropy_reg = self._compute_router_entropy_regularizer(layerwise_fusion_module)
            if router_entropy_reg is not None:
                total_loss = lm_loss + router_entropy_loss_coef * router_entropy_reg
                outputs.loss = total_loss

        self._last_loss_breakdown = {
            "lm_loss": lm_loss.detach() if lm_loss is not None else None,
            "router_entropy_reg": router_entropy_reg.detach() if router_entropy_reg is not None else None,
            "total_loss": outputs.loss.detach() if getattr(outputs, "loss", None) is not None else None,
            "router_entropy_loss_coef": router_entropy_loss_coef,
        }

        # Clear fusion context only when not training; keep tensors during train + checkpoint recompute.
        if layerwise_fusion_module is not None and (not self.training):
            layerwise_fusion_module.clear_context()

        return outputs
            
    
            

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # In generate(): forward must not wipe LayerwiseVisionFusion context mid-rollout.
        self._inputs_embeds_from_generate = True
        try:
            result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        finally:
            # End of generate(): reset flag and clear fusion context.
            self._inputs_embeds_from_generate = False
            # Clear cached ViT / router context
            layerwise_fusion_module = getattr(self.model, "layerwise_fusion", None)
            if layerwise_fusion_module is not None:
                layerwise_fusion_module.clear_context()
        
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
