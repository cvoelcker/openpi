"""TrainConfigs for the RLT-encoder + FB-objective arm.

Kept out of `config.py`, following the same lazy-import pattern as
`roboarena_config.get_roboarena_configs`. See `src/openpi/models/pi05fb_rlt.py` for the model. The
two-line splice below is the only edit this arm needs to an existing file, and it is unavoidable:
`_CONFIGS` is the registry `get_config()` reads, and `TrainConfig.freeze_filter` is
`tyro.conf.Suppress`, so a config cannot be reached by CLI override.

    import openpi.training.misc.rlt_config as rlt_config     # with the other imports

    *rlt_config.get_rlt_configs(),                           # inside _CONFIGS
"""

import openpi.models.pi05fb_rlt as pi05fb_rlt


def _shape_fields() -> dict:
    """Fields that change parameter SHAPES, and therefore must be identical between the `model=`
    and `freeze_filter=` literals. `scripts/train.py::_check_freeze_filter_matches_model` hard-
    fails at startup if the two disagree, which is exactly the guard we want -- but it means the
    two literals must be built from one dict rather than typed twice.
    """
    return {
        "pi05": True,
        "action_horizon": 10,
        "discrete_state_input": False,
        "rep_backbone_grad_scale": 0.0,
        "B_input": "state",
        "F_input": "state_action",
        # --- encoder geometry. See the cost table in pi05fb_rlt.py's module docstring. ---
        # Per block: 4*W^2 + 2*W*M, so WIDTH dominates. At W=2048/depth=2/M=4096 the two
        # encoders total ~147M trainable. Drop to W=768/depth=3/M=3072/rep_dim=768 for ~47M.
        "rlt_enc_width": 2048,
        # depth 3, matching the w768 arm, so "wide" vs "narrow" is a cleaner comparison: the two
        # arms now differ in WIDTH and d_fb only, not in depth as well. Costs one extra block per
        # encoder = 2 * 33.58M = +67.2M trainable (151.0M -> 218.2M) and ~+4.2 GB of retained
        # activations per encoder at 32/device (rlt_remat=False).
        "rlt_depth": 3,
        "rlt_mlp_dim": 4096,
        "rlt_num_heads": 16,
        # d_fb. MUST be <= min(enc_width_B, enc_width_F): rank(E[BB^T]) <= enc_width + 1, and
        # Q_r = F^T z_r only sees span(F). Pi0FBRLTConfig.__post_init__ validates this.
        "rep_dim": 2048,
        # Ortho weight is DERIVED, not set: leaving fb_ortho_coeff at None makes Pi0FBRLT resolve
        # it to 1/rep_dim. Do not pin a literal here -- it will not follow rep_dim, and the arms
        # in this file run at d=2048 and d=768.
        # False = no backward recompute in the encoder blocks. Affordable at 32/device.
        "rlt_remat": False,
    }


# Smoke-test geometry. Kept as one dict so the `model=` and `freeze_filter=` literals cannot
# drift -- `_check_freeze_filter_matches_model` hard-fails at startup if they do.
_SMOKE_SHAPES = {
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "action_horizon": 4,
    "rlt_enc_width": 32,
    "rlt_depth": 1,
    "rlt_mlp_dim": 64,
    "rlt_num_heads": 4,
    "rep_dim": 32,
}


def rlt_model(**extra):
    """`extra` OVERRIDES the shape fields, so an arm can flip one knob without a duplicate
    literal drifting out of sync with the freeze_filter copy."""
    return pi05fb_rlt.Pi0FBRLTConfig(**{**_shape_fields(), **extra})


def get_rlt_configs():
    # Imported here to avoid a circular import (config.py imports this module).
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotLiberoDataConfig
    from openpi.training.config import TrainConfig
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    def _libero_rollout_data(**base_extra):
        return LeRobotLiberoDataConfig(
            repo_id="yajat/libero90_pi05_rollouts",
            base_config=DataConfig(
                prompt_from_task=True,
                # F takes no z, so there is no goal-derived latent to build.
                include_future_observation=False,
                # Neither this model nor Pi0FB ever reads `goal_observation`; leaving it on costs
                # a full extra frame decode + transform + device transfer per sample.
                include_goal_observation=False,
                **base_extra,
            ),
            # The POLICY'S OWN norm stats, never stats recomputed from the rollouts. The rollouts
            # were produced under these; training the heads under different ones makes the frozen
            # backbone see one input distribution at train time and another at deploy time. That
            # mismatch cost 26% -> 16% once already.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
                asset_id="physical-intelligence/libero",
            ),
            extra_delta_transform=False,
        )

    return [
        TrainConfig(
            # End-to-end smoke test: dummy Gemma variants and a minimal encoder keep the compile
            # small, and training from random init proves the config / freeze-filter / data / loss
            # plumbing, nothing about learning. Same data path as the real arm, and it must point
            # at a real LeRobot repo -- rl_data_loader rejects FakeDataConfig under
            # goal-conditioned sampling. The SigLIP tower is not variant-scaled, so the prefix is
            # still 968 tokens and dominates smoke-test runtime.
            name="pi05_fb_rlt_smoke",
            model=rlt_model(**_SMOKE_SHAPES),
            freeze_filter=rlt_model(**_SMOKE_SHAPES).get_freeze_filter(),
            data=_libero_rollout_data(),
            batch_size=2,
            num_train_steps=3,
            save_interval=3,
            keep_period=3,
            log_interval=1,
            val_fraction=0.0,
            wandb_enabled=False,
            overwrite=True,
            exp_name="rlt_smoke",
            num_workers=0,
        ),
        TrainConfig(
            # The real arm. Same data, norm stats and cadence as pi05_fb_onpolicy, so "RLT
            # encoder" vs "meanpool head" is close to a single-variable comparison.
            #
            # d_fb (2048) deliberately far exceeds batch, since d bounds the dimension of the
            # reward space the method can express at all. Read the diagnostics accordingly:
            # `fb/B_ortho_err` is floored at (d-b)/d ~ 0.97 and is meaningless (use `fb/B_sigma2`
            # and `rep/B/eff_rank`), and `fb/M_rank_top1` is degenerate on TRAIN once d >= b (use
            # the val key). The ortho estimator stays unbiased but its variance scales ~d/b.
            #
            # batch_size is GLOBAL, so 256 = 32 per node across 8 GH200s -- the per-node ceiling.
            # The launcher MUST use srun: train.py calls jax.distributed.initialize() when
            # SLURM_NNODES>1, which needs SLURM_STEP_NODELIST, and that exists only inside an srun
            # step. fsdp_devices stays 1: the model fits on one GH200, so this is pure data
            # parallel and only the trainable gradients are all-reduced.
            name="pi05_fb_rlt_onpolicy",
            model=rlt_model(),
            freeze_filter=rlt_model().get_freeze_filter(),
            data=_libero_rollout_data(),
            batch_size=256,
            fsdp_devices=1,
            # 1e-4, not the 1e-3 the meanpool arms used -- that was calibrated for a ~0.4M linear
            # probe, and this is a ~147M transformer. Warmup 500 for the same reason.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-4, decay_steps=10_000, decay_lr=1e-5
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
            # RLT trains the RL token for 2000-10000 gradient steps on single-task data (App. B).
            num_train_steps=10_000,
            save_interval=2_000,
            keep_period=5_000,
            # 32 samples/node x 2 decoded frames (anchor + next) = 64 full transform passes per
            # step. Vista nodes have 72 cores and each process owns one, so give the loader room
            # -- at num_workers=4 this arm is loader-bound, not GPU-bound.
            num_workers=16,
        ),
        TrainConfig(
            # Budget arm: the ~47M configuration from the cost table, for when the 147M arm is
            # too slow to iterate on. d_fb=768 is still 24x the old rep_dim=32 and, unlike 2048,
            # is within a factor of ~6 of the batch size rather than ~32.
            name="pi05_fb_rlt_onpolicy_w768",
            model=rlt_model(rlt_enc_width=768, rlt_depth=3, rlt_mlp_dim=3072, rlt_num_heads=12, rep_dim=768),
            freeze_filter=rlt_model(
                rlt_enc_width=768,
                rlt_depth=3,
                rlt_mlp_dim=3072,
                rlt_num_heads=12,
                rep_dim=768,
            ).get_freeze_filter(),
            data=_libero_rollout_data(),
            # 64 global = 32 per node x 2 nodes (the per-node ceiling on this cluster).
            batch_size=64,
            fsdp_devices=1,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-4, decay_steps=10_000, decay_lr=1e-5
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
            num_train_steps=10_000,
            save_interval=2_000,
            keep_period=5_000,
            num_workers=16,
        ),
    ]
