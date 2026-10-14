from dataclasses import replace
from pathlib import Path

import pytest
import torch

from finetune.distillation.config import (
    Architecture,
    ExperimentConfig,
    HiddenConfig,
    LossConfig,
    TrainingConfig,
)
from finetune.distillation.models import build_model, parameter_report

CONFIGS = Path(__file__).resolve().parents[2] / "example" / "distillation"


@pytest.fixture
def config():
    return ExperimentConfig.load(CONFIGS / "smoke_cpu.yaml")


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_validate_and_inspect_without_weight_allocation(path):
    config = ExperimentConfig.load(path)
    report = parameter_report(config.student)
    assert sum(report["groups"].values()) == report["parameters"]
    assert report["mimi_included"] is False
    assert report["parameters"] < parameter_report(config.teacher)["parameters"]


def test_meta_count_matches_real_model(config):
    model = build_model(config.student)
    assert (
        sum(p.numel() for p in model.parameters())
        == parameter_report(config.student)["parameters"]
    )
    assert model.n_q == 16
    assert model.dep_q == 8
    assert model.delays == config.teacher.lm_kwargs()["delays"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dim", 0),
        ("num_layers", True),
        ("num_layers", 2.5),
        ("num_heads", 3),
        ("dim", 34),
        ("text_card", 3),
        ("hidden_scale", float("nan")),
        ("hidden_scale", 0),
    ],
)
def test_invalid_architecture_rejected(config, field, value):
    with pytest.raises(ValueError, match=r"must|Widths|Vocabulary"):
        replace(config.student, **{field: value})


def test_unknown_fields_rejected_before_model_construction(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        (CONFIGS / "smoke_cpu.yaml").read_text() + "\ninitialization: pca\n"
    )
    with pytest.raises(ValueError, match=r"Unknown.*initialization"):
        ExperimentConfig.load(path)


def test_missing_sections_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\npurpose: synthetic_smoke\n")
    with pytest.raises(ValueError, match="Missing config section"):
        ExperimentConfig.load(path)


def test_vocabulary_and_mapping_compatibility(config):
    with pytest.raises(ValueError, match="vocabularies"):
        replace(config, student=replace(config.student, card=32))
    with pytest.raises(ValueError, match="exceeds"):
        replace(config, hidden=HiddenConfig([1, 3], [2, 4]))
    with pytest.raises(ValueError, match="increasing"):
        HiddenConfig([1, 1], [2, 4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_weight": -1},
        {"audio_temperature": 0},
        {"text_temperature": float("inf")},
        {"audio_codebook_weights": [1] * 7},
        {"audio_codebook_weights": [0] * 8},
        {
            "data_weight": 0,
            "text_kl_weight": 0,
            "audio_kl_weight": 0,
            "hidden_weight": 0,
        },
    ],
)
def test_loss_config_rejects_invalid_objectives(kwargs):
    with pytest.raises(ValueError, match=r"must|required|At least"):
        LossConfig(**kwargs)


def test_smoke_resource_limits_are_explicit(config):
    with pytest.raises(ValueError, match="Smoke limits"):
        replace(config, training=TrainingConfig(steps=1001))
    with pytest.raises(ValueError, match="three frames"):
        replace(config, training=TrainingConfig(frames=1))
    with pytest.raises(ValueError, match="boolean"):
        TrainingConfig(gradient_checkpointing="false")


def test_moshi_special_tokens_survive_small_vocabulary(config):
    model = build_model(config.student)
    assert 0 <= model.text_padding_token_id < config.student.text_card
    assert model.zero_token_id < 0
    assert isinstance(config.student, Architecture)
    assert model.text_emb.weight.dtype == torch.float32
