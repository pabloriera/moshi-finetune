"""Bounded synthetic-token integration experiment using actual Moshi LMs."""

import json
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path

import moshi
import torch
import yaml
from moshi.models.loaders import get_moshi_lm
from safetensors.torch import load_file, save_file

from .config import ExperimentConfig
from .hooks import LayerCapture
from .losses import HiddenMatcher, prediction_losses
from .models import build_model, parameter_report


def _verify_reload(student, restored, codes):
    student.eval()
    restored.eval()
    with torch.no_grad():
        expected, actual = student(codes), restored(codes)
    errors = []
    for key, mask_key in (("text_logits", "text_mask"), ("logits", "mask")):
        mask = getattr(expected, mask_key)
        if not torch.equal(mask, getattr(actual, mask_key)):
            raise RuntimeError("Checkpoint changed logit validity masks")
        left, right = getattr(expected, key)[mask], getattr(actual, key)[mask]
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        errors.append(float((left - right).abs().max()) if left.numel() else 0.0)
    return max(errors)


def _export(run_dir, config, student, matcher, optimizer, codes, metrics, summary):
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()},
        run_dir / "student.safetensors",
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in matcher.state_dict().items()},
        run_dir / "matcher.safetensors",
    )
    torch.save(
        {"optimizer": optimizer.state_dict(), "steps": config.training.steps},
        run_dir / "optimizer.pt",
    )
    (run_dir / "experiment.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False)
    )
    model_config = config.student.lm_kwargs()
    model_config["moshi_name"] = "student.safetensors"
    (run_dir / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    source_metadata = distribution("moshi").read_text("direct_url.json")
    manifest = {
        "format_version": 1,
        "purpose": "synthetic_smoke",
        "initialization": "random",
        "pretrained": False,
        "teacher_checkpoint": None,
        "dataset": "synthetic shared token histories",
        "moshi_version": moshi.__version__,
        "moshi_source": json.loads(source_metadata) if source_metadata else None,
        "torch_version": str(torch.__version__),
        "model_config": model_config,
        "notes": "Toy vocabularies: not suitable for Mimi decoding or the default text tokenizer. No speech-quality claim.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )
    # Exercise upstream's actual LM loader, not only our construction helper.
    exported_config = json.loads((run_dir / "config.json").read_text())
    weights_name = exported_config.pop("moshi_name")
    restored = get_moshi_lm(
        run_dir / weights_name,
        lm_kwargs=exported_config,
        device=config.training.device,
        dtype=torch.float32,
    )
    summary["reload_max_logit_error"] = _verify_reload(student, restored, codes)
    restored_matcher = HiddenMatcher(
        config.teacher.dim, config.student.dim, len(config.hidden.student_layers)
    )
    restored_matcher.load_state_dict(
        load_file(run_dir / "matcher.safetensors"), strict=True
    )
    for key, value in matcher.state_dict().items():
        torch.testing.assert_close(
            value.cpu(), restored_matcher.state_dict()[key], rtol=0, atol=0
        )
    restored_optimizer = torch.optim.AdamW(
        list(restored.parameters()) + list(restored_matcher.parameters()),
        lr=config.training.learning_rate,
    )
    restored_optimizer.load_state_dict(
        torch.load(run_dir / "optimizer.pt", map_location="cpu", weights_only=True)[
            "optimizer"
        ]
    )
    expected_optimizer = optimizer.state_dict()
    actual_optimizer = restored_optimizer.state_dict()
    if expected_optimizer["param_groups"] != actual_optimizer["param_groups"]:
        raise RuntimeError("Optimizer checkpoint changed parameter groups")
    if expected_optimizer["state"].keys() != actual_optimizer["state"].keys():
        raise RuntimeError("Optimizer checkpoint changed parameter state keys")
    for key, state in expected_optimizer["state"].items():
        if state.keys() != actual_optimizer["state"][key].keys():
            raise RuntimeError("Optimizer checkpoint changed state fields")
        for field, value in state.items():
            actual = actual_optimizer["state"][key][field]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value.cpu(), actual.cpu(), rtol=0, atol=0)
            elif value != actual:
                raise RuntimeError("Optimizer checkpoint changed state values")
    summary["matcher_reload_verified"] = True
    summary["optimizer_reload_verified"] = True
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_smoke(config: ExperimentConfig, run_dir: str | Path):
    if config.purpose != "synthetic_smoke":
        raise ValueError(
            "Smoke accepts only purpose=synthetic_smoke; inspect production shapes instead"
        )
    reports = {
        "teacher": parameter_report(config.teacher),
        "student": parameter_report(config.student),
    }
    if any(report["parameters"] > 10_000_000 for report in reports.values()):
        raise ValueError(
            "Smoke limits each model to 10 million parameters; use inspect for target architectures"
        )
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.training.num_threads)
    # fork_rng keeps this bounded experiment from changing caller random state.
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.training.seed)
            return _run(config, run_dir, reports)
    finally:
        torch.set_num_threads(previous_threads)


def _run(config, run_dir, reports):
    device = config.training.device
    teacher = build_model(config.teacher, device=device).eval().requires_grad_(False)
    student = build_model(
        config.student,
        device=device,
        checkpointing=config.training.gradient_checkpointing,
    ).train()
    matcher = HiddenMatcher(
        config.teacher.dim, config.student.dim, len(config.hidden.student_layers)
    ).to(device)
    teacher_before = {k: p.detach().clone() for k, p in teacher.named_parameters()}
    student_before = {k: p.detach().clone() for k, p in student.named_parameters()}
    parameters = list(student.parameters()) + list(matcher.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate)
    codes = torch.randint(
        0,
        config.student.card,
        (config.training.batch_size, 17, config.training.frames),
        device=device,
    )
    codes[:, 0] = torch.randint(
        0,
        config.student.text_card,
        (config.training.batch_size, config.training.frames),
        device=device,
    )
    # Include silence/padding and explicit invalid audio IDs to exercise real masks.
    codes[:, 0, ::4] = student.text_padding_token_id
    codes[:, 1:, -1] = student.zero_token_id
    metrics = []
    with LayerCapture(
        teacher, config.hidden.teacher_layers, detach=True
    ) as teacher_capture, LayerCapture(
        student, config.hidden.student_layers
    ) as student_capture:
        with torch.no_grad():
            teacher_output = teacher(codes)
        teacher_hidden = teacher_capture.ordered()
        for step in range(config.training.steps):
            optimizer.zero_grad(set_to_none=True)
            student_capture.clear()
            student_output = student(codes)
            loss, components = prediction_losses(
                student_output,
                teacher_output,
                codes,
                config.loss,
                padding_ids={
                    student.text_padding_token_id,
                    student.end_of_text_padding_id,
                },
            )
            hidden_mask = torch.ones(
                codes.shape[0], codes.shape[2], dtype=torch.bool, device=device
            )
            hidden = loss.detach().new_zeros(())
            if config.loss.hidden_weight:
                hidden = matcher(student_capture.ordered(), teacher_hidden, hidden_mask)
                loss = loss + config.loss.hidden_weight * hidden
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite distillation loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            row = {
                "step": step + 1,
                "total": float(loss.detach()),
                "hidden": float(hidden.detach()),
                "grad_norm": float(grad_norm),
                **{k: float(v.detach()) for k, v in components.items()},
            }
            metrics.append(row)
    teacher_unchanged = all(
        torch.equal(p, teacher_before[k]) and p.grad is None
        for k, p in teacher.named_parameters()
    )
    student_changed = any(
        not torch.equal(p, student_before[k]) for k, p in student.named_parameters()
    )
    if not teacher_unchanged or not student_changed:
        raise RuntimeError("Frozen-teacher or student-update invariant failed")
    summary = {
        "purpose": "synthetic_smoke",
        "initialization": "random",
        "device": device,
        "seed": config.training.seed,
        "steps": config.training.steps,
        "parameter_reports": reports,
        "teacher_unchanged": teacher_unchanged,
        "student_changed": student_changed,
        "first_loss": metrics[0]["total"],
        "last_loss": metrics[-1]["total"],
        "quality_note": "Fixed synthetic histories and randomly initialized teacher; no audio quality or 24 GiB training claim.",
    }
    _export(run_dir, config, student, matcher, optimizer, codes, metrics, summary)
    return summary
