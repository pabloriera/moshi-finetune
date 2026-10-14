# Moshi-tiny: incremental implementation plan

## Objective

Produce a reusable, smaller Moshi checkpoint that can later be fine-tuned on
custom conversation-format datasets on a single 24 GiB GPU. Preserve Mimi,
the text tokenizer, both audio streams, eight generated codebooks, and stream
delays initially. Model quality and memory fit are measured outcomes, not
assumptions. No dataset or large training run is available yet.

Work happens in this independent nested clone, not in the parent Moshi checkout.
Fine-tune upstream revision: `2acc879fe7c48f885a18f6cc9548bccb2674d87b`.

## Working agreement

- Deliver a runnable minimum viable experiment in each iteration.
- Test a component on CPU with small tensors before involving real checkpoints.
- Use the actual Moshi LM for integration tests; toy weights establish numerical
  and architectural correctness, not speech quality.
- Keep the existing `train.py` workflow intact until the new path is validated.
- Record commands, dependency revisions, losses, memory, and unresolved limits.
- Never download a large checkpoint implicitly or start an unbounded GPU job.
- Gate progression on the acceptance criteria below. Do not implement all
  research ideas at once.

## Iteration 1: architecture and loss MVP

Deliver strict YAML configs, exact parameter inspection on the meta device,
masked cross-entropy and temperature-scaled forward-KL for text/audio, and
projected hidden-state matching. Candidate temporal transformer: 12 layers,
width 768, six 128-dimensional heads. Candidate depth transformer: four layers,
width 512, eight 64-dimensional heads. Compare a smaller three-layer/256-wide
depth variant and a temporal-only compression control.

Acceptance: invalid configs fail before allocation; identical teacher/student
distributions have zero KL; invalid NaN positions never contaminate the loss or
gradients; teacher tensors receive no gradients; zero-valid-token streams are
handled; stream weights have explicit normalization; counts include embeddings,
depth weights per codebook, and output heads.

## Iteration 2: synthetic end-to-end MVP

Deliver a bounded CPU smoke pipeline with real, tiny Moshi teacher/student LMs,
shared synthetic code histories, layer hooks, learned hidden projection, frozen
teacher, optimizer steps, and checkpoint save/reload. Record separate CE, text
KL, audio KL, hidden loss, gradient norms, and parameter counts. Include a
minimal target-model config and a compatibility manifest in exported artifacts.

Acceptance: deterministic runs; finite loss/gradients; student weights change;
teacher weights remain unchanged and have no gradients; checkpoint restores
student and projector; compare valid logits after reload; hooks are removed.
This is an integration harness, not a custom-data trainer.

## Iteration 3: token-data and offline-teacher MVP

Consume real datasets using the existing interleaver. Pre-encode Mimi tokens
with transcript timestamps; split by conversation/speaker before chunking;
preserve continuity metadata. Cache frozen teacher outputs separately so the
teacher need not be resident during student optimization. Begin with dense
logits on a short sample; add sparse logits only after testing their error.
Store selected projected hidden states rather than all 4096-wide states.

Acceptance: fingerprint teacher/tokenizer/Mimi/config/dataset; validate codebook
order, vocabulary, masks, sample IDs, and delays; reject mismatched or incomplete
caches; cached and online losses agree; data paths are user supplied.

## Iteration 4: smart-initialization MVP

First establish random initialization + CE/KL. Then test depth selection at
unchanged width. Finally implement calibration-based residual PCA and structured
attention/MLP pruning with explicit RMSNorm, residual-basis, RoPE, and depth-input
transformations. Distinguish a hidden-target PCA projector from a weight
initialization algorithm; the former does not implement the latter.

Start with a shared residual basis or explicitly include inter-layer basis
bridges. Do not rotate gated MLP neurons or RoPE head coordinates arbitrarily.
Keep whole attention heads where possible. Layer importance is evaluated on
held-out text/audio prediction and streaming behavior.

Acceptance: full-width/no-pruning transformation preserves valid logits within
numerical tolerance; every transferred parameter is audited for shape and
coverage; compressed initialization improves held-out loss over random under
the same training budget; failure is recorded, not hidden by extra training.

## Iteration 5: real distillation and 24 GiB MVP

Connect the tested cached-token path to a bounded student training CLI with
gradient accumulation, BF16, checkpointing, validation, optimizer resume, and
adapter-only/full-model export. Benchmark actual peak allocated/reserved VRAM,
wall time, and tokens/sec, including optimizer initialization and checkpointing.
Increase clip duration gradually instead of permanently training on short clips.

Acceptance: one small held-out experiment completes and resumes; memory budget
includes Mimi if online encoding is enabled; exported student loads for upstream
custom-data fine-tuning; no claim of 24 GiB fit without an actual measurement.

## Iteration 6: quality and ablation MVP

Under a fixed dataset/step/compute budget compare:

1. Random initialization + CE/KL.
2. Projected/pruned initialization + CE/KL.
3. Projected/pruned initialization + CE/KL/hidden matching.

Report text/audio held-out losses, intelligibility, voice stability, latency,
turn-taking, silence handling, and multi-minute free-running behavior. Try
on-policy GKD, reverse KL, relation matching, or intermediate-size students only
when an observed failure motivates them. Maintain a broad speech data blend for
the reusable base; custom-domain adaptation is a subsequent experiment.

## Research references

- SliceGPT: https://arxiv.org/abs/2401.15024
- Minitron: https://arxiv.org/abs/2408.11796
- TinyBERT: https://arxiv.org/abs/1909.10351
- MiniLMv2: https://arxiv.org/abs/2012.15828
- GKD: https://arxiv.org/abs/2306.13649
- MiniLLM: https://arxiv.org/abs/2306.08543

## Progress

- [x] Clone upstream; inspect trainer and environment.
- [x] Iteration 1 acceptance checks: strict configs, exact meta-device counts,
  mask-safe CE/KL and normalized projected hidden matching.
- [x] Iteration 2 integration checks: real tiny Moshi LMs on CPU, frozen teacher,
  student updates, deterministic runs, checkpointing, native loader roundtrip.
- [ ] Iteration 3 real tokens and teacher cache.
- [ ] Iteration 4 audited smart initialization.
- [ ] Iteration 5 measured training and export.
- [ ] Iteration 6 quality and ablations.

Implementation commands and measured results will be recorded in
`docs/distillation.md` as each iteration is completed.

First delivery: 45 tests pass; CPU smoke loss 8.7670 -> 5.4052 over 10 updates;
reloaded valid logits agree exactly. All initialization is random at this stage.
Native-loader, matcher, and optimizer restoration are verified, but no resume
CLI, real checkpoint distillation, PCA weight transfer, or GPU-fit measurement
is implemented yet. See [experiment guide](docs/distillation.md).
