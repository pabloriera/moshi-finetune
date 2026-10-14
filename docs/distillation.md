# Moshi-tiny experiments

The roadmap is in [DISTILLATION_PLAN.md](../DISTILLATION_PLAN.md). Iterations 1
and 2 are implemented. These experiments validate architecture, loss math,
gradient flow, and checkpoint plumbing. They do not produce a usable pretrained
speech model yet. `train.py` remains the upstream custom-data fine-tuning entry
point; the new experiment CLI is separate.

## Environment

From this repository, the prepared environment is `.venv/bin/python`.
It installs a clean upstream Moshi wheel, separate from the modified parent
checkout. To avoid reinstalling CUDA libraries on this machine, it reuses the
parent environment's site-packages through a local `.pth` file. No parent
packages were changed. This local shortcut is not needed on another machine.

Tested versions:

- Python 3.12.3; PyTorch 2.9.1+cu128.
- Moshi 0.2.13 at `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`.
- Fine-tune upstream `2acc879fe7c48f885a18f6cc9548bccb2674d87b`.
- PyYAML 6.0.3; safetensors 0.7.0; pytest 9.1.1.

For a standalone component environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-distillation.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

The requirements file pins the tested experiment runtime. Upstream `train.py`
pins PyTorch 2.6 and includes ASR/data dependencies; running it needs its full
environment and separate validation. The component environment has not been
validated for a production upstream training job.

## Runnable MVPs

Inspect the proposed 299M architecture without allocating model weights or
downloading a checkpoint:

```bash
.venv/bin/python -m finetune.distillation inspect example/distillation/tiny_300m.yaml
```

Run ten CPU updates with actual tiny Moshi models and synthetic token histories:

```bash
.venv/bin/python -m finetune.distillation smoke example/distillation/smoke_cpu.yaml \
  --run-dir runs/distillation/smoke-example
```

Use a new output directory for every experiment. Existing directories are
refused. The smoke command also rejects architecture-only profiles and models
over 10M parameters. It uses float32, a random frozen teacher, and random student
initialization. Both models see the exact same histories in all 17 streams.
It makes no network requests and does not load Mimi.

Run component and integration tests:

```bash
.venv/bin/python -m pytest tests/distillation -q
```

## Candidate shapes and memory arithmetic

Exact meta-device inspection counts include all generative-model parameters:

| Profile | Temporal layers/width/heads | Depth layers/width/heads | Parameters | BF16 weights | Approx. mixed-precision Adam state |
| --- | --- | --- | ---: | ---: | ---: |
| Official teacher | 32 / 4096 / 32 | 6 / 1024 / 16 | 7,687,729,152 | 14.32 GiB | 114.56 GiB |
| `temporal_only.yaml` | 12 / 768 / 6 | 6 / 1024 / 16 | 848,154,112 | 1.58 GiB | 12.64 GiB |
| `tiny_300m.yaml` | 12 / 768 / 6 | 4 / 512 / 8 | 299,081,216 | 0.56 GiB | 4.46 GiB |
| `tiny_200m.yaml` | 12 / 768 / 6 | 3 / 256 / 4 | 197,954,560 | 0.37 GiB | 2.95 GiB |

The last column uses 16 bytes/parameter for weights/master weights, gradients,
and Adam moments. It is an estimate, not measured VRAM. It excludes activations,
Mimi, hidden matchers, allocation overhead, and optimizer temporaries. The 299M
proposal's three hidden matchers add 9,437,184 training-only parameters. Preserve
128-dimensional temporal heads and 64-dimensional depth heads in these proposals.
The unchanged depth-transformer control helps isolate audio-depth compression.

Architecture profile training/loss values are proposed defaults, not tuned
hyperparameters or real-data training recipes. The smoke runner cannot train
those profiles. The vocabulary and delays remain those of Moshi in target
profiles; toy smoke vocabularies intentionally differ.

## Objective and layer mapping

The objective is

```text
data_weight * (text CE + audio CE)
  + text_kl_weight * temperature_text^2 * KL(teacher || student)
  + audio_kl_weight * temperature_audio^2 * KL(teacher || student)
  + hidden_weight * mean(normalized hidden-state MSE)
```

Text and audio have independent weighted-valid-token denominators. Audio weights
are applied across codebooks before normalization; doubling every audio weight
does not double the loss. Text padding is downweighted, not removed by default.
Both KL and CE select valid positions **before** softmax/cross-entropy because
Moshi fills invalid delayed-logit positions with NaNs. Empty streams return a
differentiable zero. KL teacher distributions and hidden targets are detached.

Layer endpoints are one-based temporal block outputs. Target profiles initially
match student blocks 4, 8, and 12 to teacher blocks 11, 22, and 32. This sparse
mapping is a baseline, not an importance ranking. The current hidden matcher
learns a student-to-teacher linear projection and RMS-normalizes vectors before
MSE. Its parameters are training-only and saved separately. It does not perform
PCA or transfer teacher weights. Activation hooks are scoped and removed even
after failures. A logit-only ablation sets `hidden_weight: 0`; `data_weight: 0`
also supports pure prediction distillation in the smoke harness.

## Artifacts

Each successful smoke run exports:

- `experiment.yaml`: resolved experiment configuration.
- `metrics.jsonl`: per-step CE, KL, hidden loss, total, and gradient norm.
- `student.safetensors` and `config.json`: student weights and native LM config.
- `matcher.safetensors`: learned alignment projections.
- `optimizer.pt`: optimizer states and completed-step count, loaded with
  `weights_only=True` for verification.
- `manifest.json`: random initialization, synthetic-data status, runtime version,
  and installed Moshi source provenance where available.
- `summary.json`: teacher/update invariants and reload checks.

Exports are reloaded using upstream `get_moshi_lm`; valid text and audio logits
must agree exactly in this CPU harness. Matcher weights and optimizer tensors
and parameter groups are also verified. There is no resume CLI yet. These toy
checkpoints cannot be decoded using the normal Mimi/text-tokenizer vocabulary
and must not be used as pretrained adapters or deployed speech models.

## Results recorded on 2026-10-14

- 45 tests passed, including real-Moshi integration with gradient checkpointing
  enabled and disabled, deterministic repeats, NaN masking, weighted losses,
  frozen teacher, checkpoint restoration, hook cleanup, and allocation limits.
- New component/test files passed Ruff checks and formatting.
- Verified 10-step CPU smoke: total loss **8.7670 -> 5.4052** on fixed synthetic
  histories; teacher unchanged; student updated; valid-logit reload error **0**.
- Local artifacts: `runs/distillation/smoke-verified/` (ignored by Git).
- The upstream 24 GiB probe YAML parsed through `TrainArgs.load`; no upstream
  training run was launched.

An RTX 3090 with 24,576 MiB was visible; about 20,366 MiB was already occupied at
initial inspection. All checks ran on CPU. There is no measured GPU fit or
speech-quality result yet, and decreasing synthetic loss is not evidence of
successful knowledge transfer from a pretrained Moshi.

## Upstream LoRA control and next iteration

`example/moshi_7B_24gb_probe.yaml` is a separate upstream-schema memory probe:
rank 16, batch size 1, eight accumulated microbatches, ten-second sequences,
BF16, gradient checkpointing, and adapter-only saves. Set dataset and output
paths before using it. Later increase clip duration and evaluate stopping/silence
behavior; this short-context probe is not a final adaptation recipe.

Next MVP: encode a small conversation-format dataset using the existing
interleaver and compare dense cached teacher logits with online outputs. Cache
identity must include exact checkpoint/config, tokenizer, Mimi, stream delays,
sample/chunk identities, and masks. Start with prediction-only caching; hidden
caching must declare whether targets are full teacher states or fixed PCA
projections and use a compatible matcher. PCA projection and audited weight
initialization follow after that baseline, as detailed in the roadmap.
