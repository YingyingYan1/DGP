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


import os
import warnings
import shutil
import glob

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _load_adapter_weights_from_checkpoint(model_path):
    """Load mm_projector and LayerwiseVisionFusion weights from a checkpoint directory.
    Supports mm_projector.bin or merged model*.safetensors / pytorch_model*.bin shards."""
    mm_bin = os.path.join(model_path, 'mm_projector.bin')
    if os.path.isfile(mm_bin):
        state = torch.load(mm_bin, map_location='cpu')
        return {k: v.to(torch.float16) for k, v in state.items()}

    # Load from safetensors or pytorch_model*.bin shards
    ckpt_files = glob.glob(os.path.join(model_path, 'model*.safetensors')) + \
                 glob.glob(os.path.join(model_path, 'pytorch_model*.bin'))
    if not ckpt_files:
        return None

    combined = {}
    for ckpt_file in sorted(ckpt_files):
        if ckpt_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(ckpt_file)
        else:
            state = torch.load(ckpt_file, map_location='cpu')
        for k, v in state.items():
            if 'mm_projector' in k or 'layerwise_fusion' in k:
                combined[k] = v.to(torch.float16) if v.dtype != torch.float16 else v
    return combined if combined else None


def _infer_router_hidden_dim_from_state(state_dict):
    if not state_dict:
        return None
    for k, v in state_dict.items():
        if k.endswith('router_fc1.weight'):
            return v.shape[0]
    return None


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = _load_adapter_weights_from_checkpoint(model_path)
            if mm_projector_weights is None:
                raise FileNotFoundError(
                    f"No mm_projector.bin or shard containing mm_projector / layerwise_fusion keys under {model_path}"
                )
            router_hidden_dim = getattr(cfg_pretrained, 'vision_router_hidden_dim', None)
            if router_hidden_dim is None:
                router_hidden_dim = _infer_router_hidden_dim_from_state(mm_projector_weights)
            fusion_gate_init = getattr(cfg_pretrained, 'vision_fusion_gate_init', 1.0)

            # Instantiate LayerwiseVisionFusion when enabled in config
            if getattr(cfg_pretrained, 'use_layerwise_vision_fusion', False):
                print('Initializing layerwise_fusion module...')
                from llava.model.llava_arch import LayerwiseVisionFusion
                model_inner = model.get_model()
                # Hyperparameters for LayerwiseVisionFusion
                hidden_dim = cfg_pretrained.hidden_size
                num_decoder_layers = cfg_pretrained.num_hidden_layers
                vision_hidden_dim = getattr(cfg_pretrained, 'mm_hidden_size', 1024)
                # ViT depth defaults to 24 (CLIP ViT-L/14)
                num_vision_layers = 24
                
                model_inner.layerwise_fusion = LayerwiseVisionFusion(
                    hidden_dim=hidden_dim,
                    num_vision_layers=num_vision_layers,
                    num_decoder_layers=num_decoder_layers,
                    vision_hidden_dim=vision_hidden_dim,
                    router_hidden_dim=router_hidden_dim,
                    fusion_gate_init=fusion_gate_init
                )
                print(f'  Created LayerwiseVisionFusion: hidden_dim={hidden_dim}, '
                      f'num_vision_layers={num_vision_layers}, num_decoder_layers={num_decoder_layers}')

            # Split mm_projector vs. layerwise_fusion keys (handles model.* prefix from safetensors)
            projector_weights = {}
            fusion_weights = {}
            for k, v in mm_projector_weights.items():
                if k.startswith('model.layerwise_fusion.'):
                    fusion_weights[k[len('model.layerwise_fusion.'):]] = v
                elif k.startswith('layerwise_fusion.'):
                    fusion_weights[k[len('layerwise_fusion.'):]] = v
                else:
                    pk = k if k.startswith('model.') else f'model.{k}'
                    projector_weights[pk] = v
            
            # Load mm_projector
            model.load_state_dict(projector_weights, strict=False)
            print(f'Loaded mm_projector weights with {len(projector_weights)} keys')
            
            # Load layerwise_fusion
            if fusion_weights and getattr(cfg_pretrained, 'use_layerwise_vision_fusion', False):
                model_inner = model.get_model()
                if model_inner.layerwise_fusion is not None:
                    fusion = model_inner.layerwise_fusion
                    
                    # Expected keys on module vs. checkpoint
                    model_keys = set(fusion.state_dict().keys())
                    file_keys = set(fusion_weights.keys())
                    
                    # load_state_dict
                    fusion.load_state_dict(fusion_weights, strict=False)
                    
                    # Key overlap diagnostics
                    matched_keys = model_keys & file_keys
                    missing_in_file = model_keys - file_keys
                    extra_in_file = file_keys - model_keys
                    
                    # Parameter counts
                    total_params = sum(p.numel() for p in fusion.parameters())
                    router_params = sum(p.numel() for n, p in fusion.named_parameters() if 'routers' in n)
                    expert_params = sum(p.numel() for n, p in fusion.named_parameters() if 'depth_expert' in n)
                    
                    print(f'Loaded layerwise_fusion weights:')
                    print(f'  - Keys: {len(matched_keys)}/{len(model_keys)} matched')
                    print(f'  - Total params: {total_params:,} ({total_params/1e6:.2f}M)')
                    print(f'  - Routers params: {router_params:,} ({router_params/1e6:.2f}M)')
                    print(f'  - DepthExpert params: {expert_params:,} ({expert_params/1e6:.2f}M)')
                    
                    if missing_in_file:
                        print(f'  ⚠️ Missing keys in file: {list(missing_in_file)[:3]}...')
                    if extra_in_file:
                        print(f'  ⚠️ Extra keys in file: {list(extra_in_file)[:3]}...')
                    
                    if len(matched_keys) == len(model_keys):
                        print(f'  ✅ All layerwise_fusion weights loaded successfully!')
                    else:
                        print(f'  ⚠️ Some weights may not be loaded properly')
            
            # Move layerwise_fusion to model device/dtype and register forward hooks
            if getattr(cfg_pretrained, 'use_layerwise_vision_fusion', False):
                model_inner = model.get_model()
                if model_inner.layerwise_fusion is not None:
                    # Mirror backbone device / dtype
                    model_device = next(model.parameters()).device
                    model_dtype = next(model.parameters()).dtype
                    model_inner.layerwise_fusion.to(device=model_device, dtype=model_dtype)
                    model_inner.layerwise_fusion.keep_fusion_gate_fp32()
                    print(f'  Moved layerwise_fusion to {model_device}, dtype={model_dtype}')
                    # Register hooks on decoder layers
                    model_inner._register_layerwise_hooks()
                    print(f'  Registered layerwise hooks')
        else:
            # Full SFT checkpoint (model_base is None)
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            
            # ========== LayerwiseVisionFusion for full SFT checkpoints ==========
            cfg = model.config
            if getattr(cfg, 'use_layerwise_vision_fusion', False):
                print('Initializing layerwise_fusion module for SFT model...')
                from llava.model.llava_arch import LayerwiseVisionFusion
                model_inner = model.get_model()

                # Scan checkpoint shards for layerwise_fusion tensors (also infers router_hidden_dim)
                ckpt_files = glob.glob(os.path.join(model_path, 'pytorch_model*.bin')) + \
                             glob.glob(os.path.join(model_path, 'model*.safetensors'))
                fusion_weights = {}
                for ckpt_file in ckpt_files:
                    if ckpt_file.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(ckpt_file)
                    else:
                        state_dict = torch.load(ckpt_file, map_location='cpu')

                    for k, v in state_dict.items():
                        if 'layerwise_fusion' in k:
                            new_key = k.replace('model.layerwise_fusion.', '').replace('layerwise_fusion.', '')
                            fusion_weights[new_key] = v.to(torch.float16)

                router_hidden_dim = getattr(cfg, 'vision_router_hidden_dim', None)
                if router_hidden_dim is None:
                    router_hidden_dim = _infer_router_hidden_dim_from_state(fusion_weights)
                fusion_gate_init = getattr(cfg, 'vision_fusion_gate_init', 1.0)
                
                # Hyperparameters for LayerwiseVisionFusion
                hidden_dim = cfg.hidden_size
                num_decoder_layers = cfg.num_hidden_layers
                vision_hidden_dim = getattr(cfg, 'mm_hidden_size', 1024)
                num_vision_layers = 24  # CLIP ViT-L/14
                
                # Build LayerwiseVisionFusion
                model_inner.layerwise_fusion = LayerwiseVisionFusion(
                    hidden_dim=hidden_dim,
                    num_vision_layers=num_vision_layers,
                    num_decoder_layers=num_decoder_layers,
                    vision_hidden_dim=vision_hidden_dim,
                    router_hidden_dim=router_hidden_dim,
                    fusion_gate_init=fusion_gate_init
                )
                print(f'  Created LayerwiseVisionFusion: hidden_dim={hidden_dim}, '
                      f'num_vision_layers={num_vision_layers}, num_decoder_layers={num_decoder_layers}')
                
                if fusion_weights:
                    fusion = model_inner.layerwise_fusion
                    
                    # Key overlap vs. freshly built module
                    model_keys = set(fusion.state_dict().keys())
                    file_keys = set(fusion_weights.keys())
                    matched_keys = model_keys & file_keys
                    
                    # load_state_dict
                    fusion.load_state_dict(fusion_weights, strict=False)
                    
                    # Parameter counts
                    total_params = sum(p.numel() for p in fusion.parameters())
                    router_params = sum(p.numel() for n, p in fusion.named_parameters() if 'routers' in n)
                    expert_params = sum(p.numel() for n, p in fusion.named_parameters() if 'depth_expert' in n)
                    
                    print(f'  Loaded layerwise_fusion weights from SFT checkpoint:')
                    print(f'    - Keys: {len(matched_keys)}/{len(model_keys)} matched')
                    print(f'    - Total params: {total_params:,} ({total_params/1e6:.2f}M)')
                    print(f'    - Routers params: {router_params:,} ({router_params/1e6:.2f}M)')
                    print(f'    - DepthExpert params: {expert_params:,} ({expert_params/1e6:.2f}M)')
                    
                    if len(matched_keys) == len(model_keys):
                        print(f'    ✅ All layerwise_fusion weights loaded successfully!')
                    else:
                        missing = model_keys - file_keys
                        if missing:
                            print(f'    ⚠️ Missing keys: {list(missing)[:3]}...')
                else:
                    print(f'  ⚠️ No layerwise_fusion weights found in checkpoint!')
                
                # Device / dtype alignment
                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype
                model_inner.layerwise_fusion.to(device=model_device, dtype=model_dtype)
                model_inner.layerwise_fusion.keep_fusion_gate_fp32()
                print(f'  Moved layerwise_fusion to {model_device}, dtype={model_dtype}')
                
                # Register decoder hooks
                model_inner._register_layerwise_hooks()
                print(f'  Registered layerwise hooks')
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
