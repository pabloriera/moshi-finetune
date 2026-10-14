"""Strict configs for implemented experiments; no implicit checkpoint downloads."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite_number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def construct(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class Architecture:
    dim: int
    num_layers: int
    num_heads: int
    depformer_dim: int
    depformer_num_layers: int
    depformer_num_heads: int
    text_card: int = 32000
    card: int = 2048
    context: int = 3000
    hidden_scale: float = 4.125

    def __post_init__(self):
        for f in fields(self):
            if f.name != "hidden_scale":
                positive_int(getattr(self, f.name), f.name)
        finite_number(self.hidden_scale, "hidden_scale", positive=True)
        for dim, heads in (
            (self.dim, self.num_heads),
            (self.depformer_dim, self.depformer_num_heads),
        ):
            if dim % heads or (dim // heads) % 2:
                raise ValueError(
                    "Widths must divide into heads with even RoPE head dimensions"
                )
        if self.text_card < 4 or self.card < 2:
            raise ValueError("Vocabulary must include Moshi special tokens")

    def lm_kwargs(self, *, checkpointing=False):
        # Keep stream semantics fixed. Only the implemented shape knobs vary.
        from moshi.models.loaders import _lm_kwargs

        result = dict(_lm_kwargs)
        result.update(asdict(self))
        result["depformer_dim_feedforward"] = int(
            self.hidden_scale * self.depformer_dim
        )
        result["gradient_checkpointing"] = checkpointing
        return result


@dataclass(frozen=True)
class LossConfig:
    data_weight: float = 1.0
    text_kl_weight: float = 1.0
    audio_kl_weight: float = 1.0
    hidden_weight: float = 0.1
    text_temperature: float = 1.0
    audio_temperature: float = 1.0
    text_padding_weight: float = 0.5
    audio_codebook_weights: list[float] = field(
        default_factory=lambda: [2.0] + [1.0] * 7
    )

    def __post_init__(self):
        for f in fields(self):
            if f.name != "audio_codebook_weights":
                finite_number(
                    getattr(self, f.name),
                    f.name,
                    positive=f.name.endswith("temperature"),
                )
        if (
            not isinstance(self.audio_codebook_weights, list)
            or len(self.audio_codebook_weights) != 8
        ):
            raise ValueError("Exactly eight audio codebook weights are required")
        for weight in self.audio_codebook_weights:
            finite_number(weight, "audio_codebook_weight")
        if sum(self.audio_codebook_weights) == 0:
            raise ValueError("At least one codebook weight must be positive")
        if (
            self.data_weight
            + self.text_kl_weight
            + self.audio_kl_weight
            + self.hidden_weight
            == 0
        ):
            raise ValueError("At least one loss must be enabled")


@dataclass(frozen=True)
class HiddenConfig:
    # One-based temporal block endpoints; the student learns skipped teacher blocks.
    student_layers: list[int]
    teacher_layers: list[int]

    def __post_init__(self):
        if not isinstance(self.student_layers, list) or not isinstance(
            self.teacher_layers, list
        ):
            raise ValueError("Layer endpoints must be lists")
        if not self.student_layers or len(self.student_layers) != len(
            self.teacher_layers
        ):
            raise ValueError(
                "Hidden layer mappings must be nonempty and have equal lengths"
            )
        for layers in (self.student_layers, self.teacher_layers):
            for layer in layers:
                positive_int(layer, "layer endpoint")
            if layers != sorted(set(layers)):
                raise ValueError("Layer endpoints must be strictly increasing")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    device: str = "cpu"
    steps: int = 10
    batch_size: int = 1
    frames: int = 12
    learning_rate: float = 0.001
    gradient_checkpointing: bool = False
    num_threads: int = 1

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("steps", "batch_size", "frames", "num_threads"):
            positive_int(getattr(self, key), key)
        finite_number(self.learning_rate, "learning_rate", positive=True)
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")


@dataclass(frozen=True)
class ExperimentConfig:
    version: int
    purpose: str
    teacher: Architecture
    student: Architecture
    hidden: HiddenConfig
    loss: LossConfig
    training: TrainingConfig

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Only config version 1 is supported")
        if self.purpose not in ("architecture", "synthetic_smoke"):
            raise ValueError("purpose must be architecture or synthetic_smoke")
        if (self.teacher.text_card, self.teacher.card) != (
            self.student.text_card,
            self.student.card,
        ):
            raise ValueError(
                "Teacher and student must have identical token vocabularies"
            )
        if max(self.hidden.student_layers) > self.student.num_layers:
            raise ValueError("Student hidden layer endpoint exceeds model depth")
        if max(self.hidden.teacher_layers) > self.teacher.num_layers:
            raise ValueError("Teacher hidden layer endpoint exceeds model depth")
        if self.purpose == "synthetic_smoke":
            if (
                self.training.steps > 1000
                or self.training.frames > 256
                or self.training.batch_size > 16
            ):
                raise ValueError(
                    "Smoke limits: steps <= 1000, frames <= 256, batch_size <= 16"
                )
            if self.training.frames < 3:
                raise ValueError(
                    "Smoke needs at least three frames for delayed audio predictions"
                )

    @classmethod
    def load(cls, path: str | Path):
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Experiment YAML must contain a mapping")
        values = dict(values)
        for key, kind in (
            ("teacher", Architecture),
            ("student", Architecture),
            ("hidden", HiddenConfig),
            ("loss", LossConfig),
            ("training", TrainingConfig),
        ):
            if key not in values:
                raise ValueError(f"Missing config section: {key}")
            values[key] = construct(kind, values[key])
        return construct(cls, values)
