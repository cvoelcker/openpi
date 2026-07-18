# DEBUG — pi05rep CRL representation training

Debugging the frozen-backbone contrastive-RL (CRL) rep head on LIBERO. Read before touching
`pi05rep.py` / `rl_data_loader.py`. See CLAUDE.md for the model design.

## Setup

Config `pi05_crl_libero_full_finetune_frozen`. Frozen pi05 backbone; only phi/psi rep heads
train (`action_loss_coeff=0`, `rep_backbone_grad_scale=0` ⇒ `backbone_frozen`). Two `_RepHead`s
(depth 2) attention-pool a learned softmax mix over all backbone layers → `rep_dim=2048`.
`psi`=future target (prefix, state-only); `phi`=anchor (suffix if `phi_input="state_action"`).
Symmetric InfoNCE, in-batch + one within-task negative. Run via `train_rep.py` +
`rl_data_loader.py` (`val_fraction=0.05`). **Symptom: train `rep_loss` ≪ val.**

## THE key insight (don't lose this)

**A shortcut cannot explain a train/val GAP.** The val split is *episode-level, not task-level*
(`_split_episode_indices`), so val episodes are other rollouts of the *same* tasks/prompts/scenes
seen in train. Any shortcut (prompt/scene fingerprint) transfers to val and lowers **both** curves
— it sets a floor, doesn't open a gap. Therefore:
- The gap = **genuine memorization** above the shortcut floor → capacity/regularization/data.
- **Contrastive val loss is a poor quality metric**: shortcut floors it too. Gap ⇒ memorization;
  absolute loss ⇒ mostly shortcut. To measure quality you need a shortcut-proof metric
  (task-balanced val InfoNCE, or a downstream goal-distance/success probe).

| Problem | Signature | Lever |
|---|---|---|
| Memorization | train ≪ val | ↓rep_dim, ↓depth, dropout/WD, more episodes |
| Shortcut (quality) | no gap, downstream bad | prompt handling, within-task negatives |
| Val-composition artifact | modest gap moving with val_fraction | batch composition / masking |

## Changes on this branch (`rep-head-redesign`)

Earlier session:
1. Removed blanket same-episode masking in `compute_loss` (was deleting all intra-episode hard
   negatives). `episode_id` still plumbed for windowed masking + the collision diagnostic.
2. Independent aug keys for anchor/future/negative (shared key = appearance shortcut).
3. Geometric future fix (`future_gamma=0.95`, `_sample_truncated_geometric`, inverse-CDF; no
   boundary spike; future decoupled from goal).
4. Explicit within-task negative (one per anchor, uniform over same-task frames, per-split map),
   psi-encoded, entering only the anchor→future direction. Config
   `DataConfig.include_negative_observation`.
5. Tests updated (`rl_data_loader_test.py`).

Regularizer + val-loader session (targets C1–C4 below):
1. **L2-normalize phi/psi/psi_neg + learnable temperature** (`self.logit_scale`, init
   `crl_temperature_init=0.07`, clamped `exp≤100`); rep math in float32 → InfoNCE on cosine sims.
2. **`rep_head_dropout`** on the pooled query vector, train-only, via an explicit jax key (nnx
   bridge doesn't thread flax dropout rngs). Default 0; `0.1` on the frozen config.
3. **`logsumexp_penalty_coeff`** now a knob (default `0.01`, was hardcoded `0.1`).
4. **Val-loader worker parity** (`_make_loader`): val uses `config.num_workers` too, so its
   future/goal/negative sampling diversity matches train.
5. **Diagnostics** from `compute_loss`, logged train+val: `phi_norm`/`psi_norm` (pre-norm),
   `temperature`, `phi_mix_entropy`/`psi_mix_entropy`/`*_mix_max`, `collision_rate`.
6. **Freeze filter** widened to keep temperature trainable: `.*((phi|psi)_(head|mix|proj)|logit_scale).*`.

## Status update (2026-07-13)

Findings since the sections below were written (treat older claims as superseded where they clash):
- **Ablation: same-task negatives masked from the InfoNCE denominator → gap persisted.** This rules
  out same-task near-duplicate false negatives / episode-fingerprinting as the main gap driver
  (check whether the ablation also dropped the explicit `crl_self_neg`; if not, compare train vs
  val `within_task_loss` before fully closing this).
- **Val rep_loss is FLAT while train falls; val level is high.** So the checkpoint is not
  degrading — the issue is train-side memorization on top of a high val floor. The train/val gap
  is not the right target metric; the val *floor* (and a downstream probe) is.
- Working hypothesis now: **plain long-tail instance memorization** (Zhang et al. 2017 /
  Feldman 2020) by a still-huge head (~110M params in the width-2048 gemma block even at
  depth 1 — `rep_dim` cuts don't touch the block) on frozen, deterministic features for
  ~28 epochs with effectively zero weight decay (`AdamW.weight_decay=1e-10`) and a constant LR
  after warmup. Possibly amplified in *loss* units by the learnable temperature (check where
  `temperature` settles; loss gaps at scale→100 overstate quality gaps).

### Val ranking probe (implemented)

Run 20k numbers (small-capacity run): train 0.163, val 2.026 (nomask 1.947, same-ep-masked
1.163), val collision 0.014 vs train 0.00074 (~19x), temperature 0.03, best val 1.847 @ 12k
(val creeping up → early-stop ~12k). Same-episode collisions alone are ~40% of the val level
(0.78 nats), not the earlier "20% of the gap". BUT masking same-episode pairs is over-inclusive:
it also deletes the *legitimate* discrimination CRL must learn (ordering same-episode frames by
temporal progress), so `rep_loss_masked` is only an optimistic lower bound on the achievable val
loss — the Bayes-optimal critic pays occupancy-ratio costs on same-episode futures, it doesn't
zero them.

Hence the **val ranking probe** (in `compute_loss`, val-only via `train_rep.val_step`; needs the
`*_frame_index` keys the loader now emits): scores the batch's own draws (next / future / goal /
negative) against the anchor with the phi/psi cosine and checks, per anchor:
- `rank_order_acc` / `rank_order_margin` / `rank_order_pairs` — temporally NEARER same-episode
  futures must outscore farther ones (occupancy is geometric in offset). Shortcut-proof: scene
  fingerprints are constant within an episode. **This is the "is it learning progress or just
  scene identity" metric.** Chance = 0.5.
- `rank_neg_other_acc` / `rank_neg_past_acc` (+ `_pairs`) — same-episode futures must outscore
  zero-occupancy candidates: the explicit negative when it lands in another episode / in the
  anchor's past. Chance = 0.5.
- `ranking_loss` — aggregate pairwise logistic loss at the InfoNCE logit scale.
None of these depend on batch composition/collisions, unlike raw val InfoNCE.

Caveat (found while wiring): `val_step` evaluates **EMA params** (`state.ema_params`, decay
0.999) while train loss is live params — another train/val asymmetry alongside aug/dropout.

### Randomization test (implemented, run next)

Config **`pi05_crl_libero_full_finetune_frozen_lite_randfuture`** = frozen_lite, but
`DataConfig.random_future_control=True`: the CRL positive is replaced by a **fixed** random
cross-episode frame (deterministic per anchor via `default_rng([seed, 0x5EED, t])`, stable across
epochs — fixed pairing is essential; a resampled pairing can't be memorized and the test is
vacuous). Semantics destroyed, memorizability kept.

Read-out:
- chance = ~ln(batch_size) ≈ ln 257 ≈ 5.55 nats (symmetric InfoNCE, B=256 + self-neg).
- **train rep_loss sinking below chance = pure pairing-memorization capacity**; its depth
  calibrates how much of the real run's train descent is memorization.
- val rep_loss should pin at chance (its pairings are also random).
- Caveat: control partners are drawn from the whole dataset (both splits); irrelevant for the
  memorization read-out since pairs are semantically empty. LeRobot path only (RLDS raises).

## Hypotheses / causes

Newly found (this session), ranked:
- **C1 no rep norm / no temperature** — logits were raw dot products of unbounded reps → train
  loss lowerable by inflating `‖rep‖`, val doesn't follow. **Strongest.** → fixed (L2+temp).
- **C2 inert head dropout** — Blocks ran `dropout=0`; zero head regularization. → fixed.
- **C3 val-loader RNG asymmetry** (metric confound) — val `num_workers=0`, one persistent RNG vs
  train's N → lower-variance val sampling. → fixed (worker parity).
- **C4 tiny val within-task negative pool** (~5%, split-restricted) → val difficulty differs.
  Left as-is (in-split is correct semantics); watch via `collision_rate`.
- **C5 aug asymmetry** — train augments, val doesn't → true overfitting *worse* than shown;
  train/val loss not directly comparable.

Prior list (still open):
- **H1 head capacity/memorization** (primary for the gap) — sweep `rep_dim∈{256,512}`,
  `rep_head_depth=1`; does the gap shrink?
- **H3 val same-episode collision** — modest gap that shrinks as `val_fraction↑`; lever = windowed
  same-episode mask or task-aware batching.
- **H4 shortcut reliance** (no gap; hurts downstream) — needs task-balanced/probe metric.
- **H5 contrastive collapse** — rep std→0, loss→log(B); watch `phi_norm`/mix stats.
- **H6 layer-mix collapse** — `*_mix_entropy`/`*_mix_max` diagnostics now track this; lever =
  entropy reg or restrict to early layers.
- **H8 same-task goal-collision false negatives** — floor above 0; lever = mask same-task pairs
  only when both near-goal/temporally close.
- **H9 norm_stats mismatch** (memory `rep-action-loss-norm-stats-gotcha`) — check norm stats first
  if action/state inputs look off.
- **H10 phi/psi asymmetry** — `state_action` phi reads suffix (noisy actions) while psi reads
  prefix; try `phi_input="state"`.

## Verify (macOS box is CUDA-locked; run on training box)

```
uv run pytest src/openpi/training/rl_data_loader_test.py -q
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_rep.py \
  pi05_crl_libero_full_finetune_frozen --exp-name=dbg-reg --overwrite
```
Watch: train vs val `rep_loss` gap; `phi_norm`/`psi_norm` bounded; `temperature` settles; mix
entropy not collapsing; train vs val `collision_rate`. No `pi05rep_test.py` yet — add one
asserting unit-norm reps and that the loss matches pre-change at dropout=0/temp=1/penalty=0.1.

## TODO

- `episode_id` plumbed but unused in loss (kept for windowed masking H3/H8).
- Add a shortcut-proof quality metric (task-balanced val InfoNCE or downstream probe).
- Non-frozen configs run the 3B forward if `include_negative` is on (extra compute, discarded).
