import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from finetune.distillation.config import ExperimentConfig
from finetune.distillation.hooks import LayerCapture
from finetune.distillation.models import build_model
from finetune.distillation.smoke import run_smoke

CONFIGS = Path(__file__).resolve().parents[2] / "example" / "distillation"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_moshi_forward_backward_and_checkpoint_reload(tmp_path, checkpointing):
    config = ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")
    config = replace(
        config,
        training=replace(
            config.training, steps=3, gradient_checkpointing=checkpointing
        ),
    )
    summary = run_smoke(config, tmp_path / "run")
    assert summary["teacher_unchanged"]
    assert summary["student_changed"]
    assert summary["reload_max_logit_error"] == 0
    assert summary["matcher_reload_verified"]
    assert summary["optimizer_reload_verified"]
    metrics = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["grad_norm"] > 0 and row["hidden"] > 0 for row in metrics)
    assert metrics[-1]["total"] < metrics[0]["total"]
    assert ExperimentConfig.load(tmp_path / "run" / "experiment.yaml") == config
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["pretrained"] is False
    assert manifest["model_config"]["moshi_name"] == "student.safetensors"


def test_smoke_is_deterministic_and_preserves_caller_rng_and_threads(tmp_path):
    config = ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")
    config = replace(config, training=replace(config.training, steps=2))
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    first = run_smoke(config, tmp_path / "first")
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    second = run_smoke(config, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first" / "metrics.jsonl").read_text() == (
        tmp_path / "second" / "metrics.jsonl"
    ).read_text()


def test_hooks_removed_even_after_failure():
    config = ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")
    model = build_model(config.student)
    capture = LayerCapture(model, [1, 2])
    counts = [len(layer._forward_hooks) for layer in model.transformer.layers]
    with pytest.raises(RuntimeError, match="simulated"):
        with capture:
            raise RuntimeError("simulated failure")
    assert [len(layer._forward_hooks) for layer in model.transformer.layers] == counts
    assert capture.handles == []
    assert capture.states == {}


def test_production_architecture_cannot_launch_through_smoke(tmp_path):
    config = ExperimentConfig.load(CONFIGS / "tiny_300m.yaml")
    with pytest.raises(ValueError, match="purpose=synthetic_smoke"):
        run_smoke(config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_large_models_cannot_bypass_smoke_allocation_limit(tmp_path):
    config = ExperimentConfig.load(CONFIGS / "tiny_300m.yaml")
    config = replace(config, purpose="synthetic_smoke")
    with pytest.raises(ValueError, match="10 million"):
        run_smoke(config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_logit_only_ablation_without_hidden_objective(tmp_path):
    config = ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")
    config = replace(
        config,
        loss=replace(config.loss, hidden_weight=0, data_weight=0),
        training=replace(config.training, steps=2),
    )
    summary = run_smoke(config, tmp_path / "run")
    assert summary["teacher_unchanged"]
    assert summary["student_changed"]
    metrics = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(
        row["hidden"] == 0 and row["text_ce"] == 0 and row["audio_ce"] == 0
        for row in metrics
    )


def test_existing_output_directory_is_never_overwritten(tmp_path):
    config = ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")
    marker = tmp_path / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        run_smoke(config, tmp_path)
    assert marker.read_text() == "existing data"
