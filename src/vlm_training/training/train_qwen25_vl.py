from __future__ import annotations

import csv
import inspect
import json
import logging
import os
import time
from importlib import metadata as importlib_metadata
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import hydra
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, TrainerCallback

from ..data.data import load_qwen_json_dataset, to_trl_messages
from ..data.resolver import prepare_training_dataset
from ..runtime.accelerator import prepare_runtime

logger = logging.getLogger(__name__)


class CSVMetricsCallback(TrainerCallback):
    def __init__(self, output_path: Path, jsonl_path: Path, interval: int = 50):
        self.path = output_path
        self.jsonl_path = jsonl_path
        self.interval = interval
        self.header_written = False
        self._fieldnames: list[str] = []
        self._last_log_step = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if state.global_step - self._last_log_step >= self.interval:
            row = self._build_row(state, logs, "log")
            self._write_row(row)
            self._last_log_step = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        row = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else "",
            "event": "eval",
        }
        for key, value in metrics.items():
            row[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
        self._write_row(row)

    def on_epoch_end(self, args, state, control, **kwargs):
        row = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else "",
            "event": "epoch_end",
        }
        self._write_row(row)

    def _build_row(self, state, logs: dict, event: str) -> dict:
        row = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else "",
            "event": event,
        }
        for key, value in logs.items():
            row[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
        return row

    def _write_row(self, row: dict) -> None:
        keys = list(row.keys())
        if set(keys) != set(self._fieldnames):
            self._fieldnames = keys
            self.header_written = False
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class TqdmProgressCallback(TrainerCallback):
    def __init__(self):
        self._bar: tqdm.tqdm | None = None
        self._latest_metrics: dict[str, str] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self._bar = tqdm.tqdm(
            total=state.max_steps,
            desc="train",
            unit="step",
            leave=True,
            dynamic_ncols=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.update(1)
            if self._latest_metrics:
                self._bar.set_postfix(self._latest_metrics, refresh=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._bar is not None and logs:
            self._latest_metrics = {
                key: (f"{value:.4f}" if isinstance(value, float) else str(value))
                for key, value in logs.items()
                if key in {"loss", "eval_loss", "learning_rate", "grad_norm"}
            }
            if state.epoch is not None:
                self._latest_metrics["epoch"] = f"{state.epoch:.2f}"
            self._bar.set_postfix(self._latest_metrics, refresh=False)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            epoch = int(state.epoch) if state.epoch is not None else "?"
            tqdm.tqdm.write(f"--- Epoch {epoch} / {args.num_train_epochs} ---")
            self._bar.close()
            remaining = state.max_steps - state.global_step
            if remaining > 0:
                self._bar = tqdm.tqdm(
                    total=state.max_steps,
                    initial=state.global_step,
                    desc="train",
                    unit="step",
                    leave=True,
                    dynamic_ncols=True,
                )
                if self._latest_metrics:
                    self._bar.set_postfix(self._latest_metrics, refresh=False)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._bar is not None and metrics:
            parts = [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()]
            tqdm.tqdm.write("")
            tqdm.tqdm.write(f"  Eval: {', '.join(parts)}")

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


@dataclass
class Qwen2_5_VLDataCollator:
    processor: AutoProcessor
    pad_to_multiple_of: int | None = None
    assistant_start_ids: list[int] | None = None
    im_end_id: int | None = None

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[object] = []
        for example in examples:
            text = example.get("text")
            if text is None:
                text = self.processor.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
            image_value = example.get("image")
            if image_value is None:
                continue
            if isinstance(image_value, list):
                images.extend(image_value)
            else:
                images.append(image_value)

        batch = self.processor(
            text=texts,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels = batch["input_ids"].clone()
        labels[:, :] = -100

        assistant_start_ids = self.assistant_start_ids
        if assistant_start_ids is None:
            assistant_start_ids = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            self.assistant_start_ids = assistant_start_ids
        im_end_id = self.im_end_id
        if im_end_id is None:
            im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            self.im_end_id = im_end_id

        for row_index in range(len(labels)):
            label_seq = batch["input_ids"][row_index].tolist()
            token_index = 0
            while token_index < len(label_seq):
                if label_seq[token_index : token_index + len(assistant_start_ids)] == assistant_start_ids:
                    token_index += len(assistant_start_ids)
                    start_idx = token_index
                    while token_index < len(label_seq) and label_seq[token_index] != im_end_id:
                        token_index += 1
                    end_idx = min(token_index + 1, len(label_seq))
                    labels[row_index, start_idx:end_idx] = batch["input_ids"][row_index, start_idx:end_idx]
                else:
                    token_index += 1

        mask_token_ids = set()
        if self.processor.tokenizer.pad_token_id is not None:
            mask_token_ids.add(self.processor.tokenizer.pad_token_id)
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_pad_id is not None:
            mask_token_ids.add(image_pad_id)
        for token_id in mask_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch


class MemoryTelemetryCallback(TrainerCallback):
    def __init__(self, jsonl_path: Path, interval: int = 25, log_first_n_steps: int = 0):
        self.jsonl_path = jsonl_path
        self.interval = max(int(interval), 1)
        self.log_first_n_steps = max(int(log_first_n_steps), 0)
        self._step_started_at: float | None = None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_started_at = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        should_log = state.global_step <= self.log_first_n_steps or state.global_step % self.interval == 0
        if state.global_step == 0 or not should_log:
            return

        row = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            "train_batch_size": getattr(args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
        }
        if self._step_started_at is not None:
            row["optimizer_step_seconds"] = round(time.perf_counter() - self._step_started_at, 4)
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            row.update(
                {
                    "cuda_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
                    "cuda_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
                    "cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
                    "cuda_free_gb": round(free_bytes / (1024**3), 4),
                    "cuda_total_gb": round(total_bytes / (1024**3), 4),
                }
            )

        try:
            import psutil  # type: ignore
        except Exception:
            psutil = None
        if psutil is not None:
            vm = psutil.virtual_memory()
            row.update(
                {
                    "ram_available_gb": round(vm.available / (1024**3), 4),
                    "ram_total_gb": round(vm.total / (1024**3), 4),
                    "ram_used_percent": vm.percent,
                }
            )

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logger.info("Logging to %s", log_path)


def _write_run_metadata(output_dir: Path, cfg: DictConfig, runtime_plan) -> None:
    metadata = {
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "runtime_plan": None if runtime_plan is None else {
            "state": runtime_plan.state,
            "attention_backend": runtime_plan.attention_backend,
            "attention_reason": runtime_plan.attention_reason,
            "use_liger_kernel": runtime_plan.use_liger_kernel,
            "liger_reason": runtime_plan.liger_reason,
            "torch_compile_enabled": runtime_plan.torch_compile_enabled,
            "torch_compile_reason": runtime_plan.torch_compile_reason,
            "tf32_enabled": runtime_plan.tf32_enabled,
            "bf16_enabled": runtime_plan.bf16_enabled,
            "fp16_enabled": runtime_plan.fp16_enabled,
            "gpu": None if runtime_plan.gpu is None else asdict(runtime_plan.gpu),
            "warnings": runtime_plan.warnings,
            "install_actions": runtime_plan.install_actions,
        },
    }
    (output_dir / "resolved_training_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _quantization_config(cfg: DictConfig) -> BitsAndBytesConfig | None:
    if not cfg.quantization.enabled:
        return None
    if cfg.quantization.bits != 4:
        raise ValueError("Only 4-bit QLoRA is supported for this training recipe.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.compute_dtype),
    )


def _resolve_lora_target_modules(cfg: DictConfig, model) -> list[str]:
    configured_targets = list(cfg.lora.target_modules)
    excluded_keywords = [str(keyword).lower() for keyword in _get(cfg.lora, "exclude_module_name_keywords", default=[])]
    if not excluded_keywords:
        return configured_targets

    resolved_targets: list[str] = []
    configured_target_set = set(configured_targets)
    for module_name, _module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in configured_target_set:
            continue
        if any(keyword in module_name.lower() for keyword in excluded_keywords):
            continue
        resolved_targets.append(module_name)

    if not resolved_targets:
        logger.warning("No LoRA target modules remained after exclusions. Falling back to configured targets: %s", configured_targets)
        return configured_targets

    logger.info(
        "Resolved %d concrete LoRA target modules after excluding keywords: %s",
        len(resolved_targets),
        excluded_keywords,
    )
    return resolved_targets


def _lora_config(cfg: DictConfig, model=None):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        target_modules=list(cfg.lora.target_modules) if model is None else _resolve_lora_target_modules(cfg, model),
        task_type="CAUSAL_LM",
    )


def _get(cfg: DictConfig, *keys: str, default=None):
    current = cfg
    for key in keys:
        if not isinstance(current, DictConfig) or key not in current:
            return default
        current = current[key]
    return current


def _load_trl_runtime():
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "trl is required for training but is not installed in the active environment. "
            "Install project dependencies first, for example: `pip install -e .`"
        ) from exc
    return SFTConfig, SFTTrainer


def _patch_transformers_flash_attn_mapping() -> None:
    try:
        from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    except Exception:
        return

    if "flash_attn" in PACKAGE_DISTRIBUTION_MAPPING:
        return

    candidates = []
    for dist_name in ("flash-attn", "flash_attn"):
        try:
            importlib_metadata.version(dist_name)
            candidates.append(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue

    if candidates:
        PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] = tuple(candidates)
        logger.info("Patched transformers flash_attn package mapping: %s", candidates)


def _supported_kwargs(callable_obj, kwargs: dict) -> dict:
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def _estimate_total_training_steps(cfg: DictConfig, train_dataset_size: int) -> int:
    max_steps = int(_get(cfg.training, "max_steps", default=-1) or -1)
    if max_steps > 0:
        return max_steps
    micro_batch = max(int(cfg.training.per_device_train_batch_size), 1)
    grad_accum = max(int(cfg.training.gradient_accumulation_steps), 1)
    steps_per_epoch = max(ceil(train_dataset_size / micro_batch / grad_accum), 1)
    return max(ceil(steps_per_epoch * float(cfg.training.num_train_epochs)), 1)


def _build_sft_config(cfg: DictConfig, output_dir: Path, has_eval: bool, train_dataset_size: int):
    SFTConfig, _ = _load_trl_runtime()
    warmup_steps = _get(cfg.training, "warmup_steps", default=None)
    warmup_ratio = _get(cfg.training, "warmup_ratio", default=None)
    if warmup_steps is None and warmup_ratio is not None:
        total_steps = _estimate_total_training_steps(cfg, train_dataset_size)
        warmup_steps = max(int(total_steps * float(warmup_ratio)), 0)

    dataloader_num_workers = int(_get(cfg.training, "dataloader_num_workers", default=4) or 0)
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "max_steps": _get(cfg.training, "max_steps", default=-1),
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if cfg.training.gradient_checkpointing else None,
        "learning_rate": cfg.training.learning_rate,
        "warmup_steps": 0 if warmup_steps is None else warmup_steps,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "weight_decay": cfg.training.weight_decay,
        "optim": _get(cfg.training, "optim", default="paged_adamw_8bit"),
        "max_grad_norm": _get(cfg.training, "max_grad_norm", default=1.0),
        "bf16": cfg.training.bf16,
        "fp16": cfg.training.fp16,
        "tf32": _get(cfg.training, "tf32", default=True),
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "save_total_limit": _get(cfg.training, "save_total_limit", default=3),
        "eval_strategy": _get(cfg.training, "eval_strategy", default="steps") if has_eval else "no",
        "eval_steps": cfg.training.eval_steps,
        "report_to": list(cfg.training.report_to),
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "load_best_model_at_end": _get(cfg.training, "load_best_model_at_end", default=False),
        "metric_for_best_model": _get(cfg.training, "metric_for_best_model", default=None),
        "greater_is_better": _get(cfg.training, "greater_is_better", default=None),
        "save_strategy": _get(cfg.training, "save_strategy", default="steps"),
        "save_only_model": _get(cfg.training, "save_only_model", default=False),
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": _get(cfg.training, "dataloader_pin_memory", default=True),
        "dataloader_persistent_workers": bool(_get(cfg.training, "dataloader_persistent_workers", default=True)) if dataloader_num_workers > 0 else False,
        "dataloader_prefetch_factor": _get(cfg.training, "dataloader_prefetch_factor", default=2) if dataloader_num_workers > 0 else None,
        "auto_find_batch_size": _get(cfg.training, "auto_find_batch_size", default=False),
        "torch_compile": _get(cfg.training, "torch_compile", default=False),
        "torch_compile_backend": _get(cfg.training, "torch_compile_backend", default=None),
        "torch_compile_mode": _get(cfg.training, "torch_compile_mode", default=None),
        "use_liger_kernel": _get(cfg.training, "use_liger_kernel", default=False),
        "liger_kernel_config": _get(cfg.training, "liger_kernel_config", default=None),
        "pad_to_multiple_of": _get(cfg.training, "pad_to_multiple_of", default=8),
        "max_length": _get(cfg.training, "max_length", default=None),
        "max_seq_length": _get(cfg.training, "max_seq_length", default=4096),
    }
    return SFTConfig(**_supported_kwargs(SFTConfig.__init__, kwargs))


def _add_chat_text_column(dataset, processor):
    def _format(example: dict) -> dict:
        return {
            "text": processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    return dataset.map(
        _format,
        desc="Precomputing chat templates",
    )


@hydra.main(version_base=None, config_path="../../../configs/vlm_training", config_name="qwen25_vl_3b")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime_plan = None
    if bool(_get(cfg, "runtime", "auto_prepare", default=True)):
        runtime_plan = prepare_runtime(cfg, output_dir / "runtime_plan.json")
        logger.info(
            "Runtime plan: state=%s, attention=%s, liger=%s, compile=%s, tf32=%s, workers=%s",
            runtime_plan.state,
            runtime_plan.attention_backend,
            runtime_plan.use_liger_kernel,
            runtime_plan.torch_compile_enabled,
            runtime_plan.tf32_enabled,
            runtime_plan.dataloader_num_workers,
        )
        for warning in runtime_plan.warnings:
            logger.warning("Runtime fallback: %s", warning)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    _write_run_metadata(output_dir, cfg, runtime_plan)

    quantization_config = _quantization_config(cfg)
    _patch_transformers_flash_attn_mapping()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
        quantization_config=quantization_config,
        device_map=cfg.model.device_map,
    )
    logger.info("Using attention backend: %s", cfg.model.attn_implementation)
    model.config.use_cache = False

    if quantization_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.training.gradient_checkpointing else None,
        )

    processor = AutoProcessor.from_pretrained(
        cfg.model.name_or_path,
        min_pixels=cfg.model.min_pixels,
        max_pixels=cfg.model.max_pixels,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prepared_dataset = prepare_training_dataset(cfg.data)
    logger.info("Resolved dataset version %s at %s", prepared_dataset.dataset_version, prepared_dataset.final_root)

    raw_train = load_qwen_json_dataset(prepared_dataset.train_json)
    dataset_num_proc = int(_get(cfg.training, "dataset_num_proc", default=1) or 1)
    train_dataset = raw_train.map(
        to_trl_messages,
        remove_columns=[column for column in raw_train.column_names if column != "image"],
        num_proc=dataset_num_proc if dataset_num_proc > 1 else None,
    )
    train_dataset = _add_chat_text_column(train_dataset, processor)

    eval_dataset = None
    if prepared_dataset.val_json:
        raw_eval = load_qwen_json_dataset(prepared_dataset.val_json)
        eval_dataset = raw_eval.map(
            to_trl_messages,
            remove_columns=[column for column in raw_eval.column_names if column != "image"],
            num_proc=dataset_num_proc if dataset_num_proc > 1 else None,
        )
        eval_dataset = _add_chat_text_column(eval_dataset, processor)

    logger.info(
        "Training flow: train_examples=%d, eval_examples=%s, micro_batch=%s, grad_accum=%s, effective_batch=%s, estimated_optimizer_steps=%d",
        len(train_dataset),
        None if eval_dataset is None else len(eval_dataset),
        cfg.training.per_device_train_batch_size,
        cfg.training.gradient_accumulation_steps,
        int(cfg.training.per_device_train_batch_size) * int(cfg.training.gradient_accumulation_steps),
        _estimate_total_training_steps(cfg, len(train_dataset)),
    )

    args = _build_sft_config(cfg, output_dir, has_eval=eval_dataset is not None, train_dataset_size=len(train_dataset))
    csv_interval = _get(cfg.training, "csv_log_interval", default=cfg.training.logging_steps * 5)
    callbacks: list[TrainerCallback] = [
        CSVMetricsCallback(
            output_dir / "training_metrics.csv",
            output_dir / "training_metrics.jsonl",
            interval=csv_interval,
        ),
        TqdmProgressCallback(),
    ]
    if bool(_get(cfg.training, "memory_telemetry", "enabled", default=True)):
        callbacks.append(
            MemoryTelemetryCallback(
                output_dir / "memory_telemetry.jsonl",
                interval=int(_get(cfg.training, "memory_telemetry", "interval", default=max(int(cfg.training.logging_steps), 1))),
                log_first_n_steps=int(_get(cfg.training, "memory_telemetry", "log_first_n_steps", default=0)),
            )
        )

    _, SFTTrainer = _load_trl_runtime()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=_lora_config(cfg, model),
        data_collator=Qwen2_5_VLDataCollator(
            processor,
            pad_to_multiple_of=_get(cfg.training, "pad_to_multiple_of", default=8),
        ),
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=_get(cfg.training, "resume_from_checkpoint", default=None) or None)
    logger.info("Training completed. Saving final adapter...")
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "processor"))

    train_metrics = train_result.metrics
    train_metrics["train_steps"] = train_result.global_step
    with (output_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(train_metrics.items()):
            writer.writerow([key, f"{value:.6f}" if isinstance(value, float) else str(value)])
    logger.info("Final training metrics saved to final_metrics.csv")


if __name__ == "__main__":
    main()
