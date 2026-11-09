import os
import torch
import torch.nn as nn

try:
    import torch.distributed as dist  # type: ignore
except ImportError:
    dist = None

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Dict, List, Optional

# Master switch: set LLAVA_SILENT_LOGS=1 to silence print-based debug output in this file.
_LLAVA_SILENT_LOGS = os.environ.get("LLAVA_SILENT_LOGS", "0") == "1"
if _LLAVA_SILENT_LOGS:
    import builtins as _bt
    def print(*args, **kwargs):  # type: ignore
        return None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# def gather_module_state_dict(module: torch.nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
#     if module is None:
#         return {}

#     params = list(module.parameters())
#     zero3_enabled = any(hasattr(p, "ds_id") for p in params)
#     state_dict: Dict[str, torch.Tensor] = {}

#     dist_available = dist is not None and dist.is_available()
#     world_rank = dist.get_rank() if dist_available and dist.is_initialized() else 0

#     if zero3_enabled and params:
#         try:
#             from deepspeed import zero
#         except ImportError:
#             zero3_enabled = False

#     if zero3_enabled and params:
#         from deepspeed import zero
#         with zero.GatheredParameters(params, modifier_rank=0):
#             should_collect = True
#             if dist_available and dist.is_initialized():
#                 should_collect = world_rank == 0
#             if should_collect:
#                 state_dict = {
#                     prefix + k: v.detach().cpu()
#                     for k, v in module.state_dict().items()
#                 }
#         if dist_available and dist.is_initialized():
#             dist.barrier()
#             if world_rank != 0:
#                 state_dict = {}
#     else:
#         state_dict = {
#             prefix + k: v.detach().cpu()
#             for k, v in module.state_dict().items()
#         }

#     return state_dict

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
        from deepspeed import zero
        
        # =========================================================
        # [FIX ADDED HERE] Force-clear active_sub_modules counters
        # Must run before entering the GatheredParameters context
        # =========================================================
        for p in params:
            if hasattr(p, 'ds_active_sub_modules'):
                p.ds_active_sub_modules.clear()
        # =========================================================

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

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            fusion_gate_parameters = {
                name for name, _ in opt_model.named_parameters()
                if "layerwise_fusion.routers." in name and name.endswith("fusion_gate_logit")
            }
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in fusion_gate_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            fusion_gate_params = [
                p for n, p in opt_model.named_parameters()
                if n in fusion_gate_parameters and p.requires_grad
            ]
            if fusion_gate_params:
                fusion_gate_group = {
                    "params": fusion_gate_params,
                    "weight_decay": 0.0,
                }
                if getattr(self.args, "fusion_gate_lr", None) is not None:
                    fusion_gate_group["lr"] = self.args.fusion_gate_lr
                optimizer_grouped_parameters.append(fusion_gate_group)

            # Ensure layerwise_fusion (router) params are included even when tune_mm_mlp_adapter=False.
            existing_param_ids = {id(p) for group in optimizer_grouped_parameters for p in group["params"]}
            router_decay = []
            router_nodecay = []
            for n, p in opt_model.named_parameters():
                if "layerwise_fusion" not in n:
                    continue
                if not p.requires_grad:
                    continue
                if id(p) in existing_param_ids:
                    continue
                if n in decay_parameters and "bias" not in n:
                    router_decay.append(p)
                else:
                    router_nodecay.append(p)
            if router_decay:
                optimizer_grouped_parameters.append({
                    "params": router_decay,
                    "weight_decay": self.args.weight_decay,
                })
            if router_nodecay:
                optimizer_grouped_parameters.append({
                    "params": router_nodecay,
                    "weight_decay": 0.0,
                })

            if os.environ.get("LLAVA_DEBUG_FUSION_GATE_OPT", "0") == "1":
                name_by_id = {id(p): n for n, p in opt_model.named_parameters()}
                opt_gate_names = []
                for gi, group in enumerate(optimizer_grouped_parameters):
                    for p in group["params"]:
                        n = name_by_id.get(id(p))
                        if n in fusion_gate_parameters:
                            opt_gate_names.append((n, gi, group.get("lr", None), group.get("weight_decay", None)))
                if self.args.local_rank in (-1, 0):
                    print("\n[FusionGateOptDebug] pre-build optimizer check")
                    print(f"[FusionGateOptDebug] discovered_gate_names={sorted(fusion_gate_parameters)}")
                    print(f"[FusionGateOptDebug] gate_params_in_groups={opt_gate_names}")
                missing_gate_names = sorted(fusion_gate_parameters - {n for n, _, _, _ in opt_gate_names})
                if missing_gate_names:
                    raise RuntimeError(
                        "Fusion gate params missing from optimizer_grouped_parameters: "
                        + ", ".join(missing_gate_names)
                    )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            ds_model = self.model.module if hasattr(self.model, "module") else self.model
            core_model = ds_model.get_model() if hasattr(ds_model, "get_model") else ds_model

            weight_to_save: Dict[str, torch.Tensor] = {}
            weight_to_save.update(
                gather_module_state_dict(getattr(core_model, "mm_projector", None), "mm_projector.")
            )

            if getattr(core_model, "layerwise_fusion", None) is not None:
                weight_to_save.update(
                    gather_module_state_dict(core_model.layerwise_fusion, "layerwise_fusion.")
                )

            if getattr(self.args, "use_im_start_end", False):
                for key in ('embed_tokens', 'embed_in'):
                    module = getattr(core_model, key, None)
                    if module is not None:
                        weight_to_save.update(
                            gather_module_state_dict(module, f"{key}.")
                        )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)















