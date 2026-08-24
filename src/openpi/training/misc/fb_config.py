"""TrainConfigs for the Forward-Backward arm (see src/openpi/models/pi05fb.py).

Kept out of `config.py`, following the same lazy-import pattern as
`roboarena_config.get_roboarena_configs`. To register, in `src/openpi/training/config.py`:

    import openpi.training.misc.fb_config as fb_config     # with the other imports

    *fb_config.get_fb_configs(),                           # inside _CONFIGS
"""

import pathlib

from openpi.models import pi0_config


def _shape_fields() -> dict:
    """Fields shared between the `model=` and `freeze_filter=` literals.

    `scripts/train.py::_check_freeze_filter_matches_model` hard-fails at startup if the two
    disagree, and `freeze_filter` is tyro.conf.Suppress so it cannot be fixed from the CLI.
    """
    return {
        "pi05": True,
        "action_horizon": 10,
        "discrete_state_input": False,
        "rep_dim": 32,
        "rep_backbone_grad_scale": 0.0,
        "rep_head_kind": "meanpool",
        "rep_head_depth": 0,
        "B_input": "state",
        "F_input": "state_action",
        "include_proprio": False,
        "F_include_prefix": False,
    }


def _fb_model(**extra):
    """`extra` overrides `_shape_fields()`, so an arm can flip one knob without a duplicate
    literal drifting out of sync with the freeze_filter copy."""
    return pi0_config.Pi0FBConfig(**{**_shape_fields(), **extra})


def get_fb_configs():
    # Imported here to avoid a circular import (config.py imports this module).
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotLiberoDataConfig
    from openpi.training.config import TrainConfig
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    def _rollout_data():
        return LeRobotLiberoDataConfig(
            repo_id="yajat/libero90_pi05_rollouts",
            # F takes no z, so there is no goal-derived latent to build.
            base_config=DataConfig(prompt_from_task=True, include_future_observation=False),
            # The policy's OWN norm stats, never stats recomputed from the rollouts: the rollouts
            # were produced under these, and the mismatch cost 26% -> 16% once.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
                asset_id="physical-intelligence/libero",
            ),
            extra_delta_transform=False,
        )

    def _schedule():
        return _optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-3, decay_steps=8_000, decay_lr=1e-4)

    return [
        TrainConfig(
            # Fast-compiling wiring check: dummy Gemma variants train from random init and only
            # prove the config/freeze-filter/data plumbing. It must still point at a REAL LeRobot
            # repo -- requires_goal_data is True, and rl_data_loader rejects FakeDataConfig under
            # goal-conditioned sampling.
            name="pi05_fb_debug_dummy",
            model=_fb_model(
                paligemma_variant="dummy",
                action_expert_variant="dummy",
                rep_head_dropout=0.1,
                fb_target_ema=0.99,
            ),
            freeze_filter=_fb_model(
                paligemma_variant="dummy",
                action_expert_variant="dummy",
            ).get_freeze_filter(),
            data=LeRobotLiberoDataConfig(
                repo_id="jesbu1/libero_90_lerobot",
                assets=AssetsConfig(
                    assets_dir=str(
                        (pathlib.Path("./assets") / "pi05_autograd_guidance_libero_full_finetune_frozen").resolve()
                    )
                ),
                base_config=DataConfig(prompt_from_task=True, repo_revisions={"jesbu1/libero_90_lerobot": "main"}),
                extra_delta_transform=False,
            ),
            batch_size=2,
            num_train_steps=10,
            save_interval=10,
            keep_period=10,
            log_interval=1,
            val_fraction=0.0,
            wandb_enabled=False,
            overwrite=True,
            exp_name="fb_debug_dummy",
        ),
        TrainConfig(
            # The real arm.
            name="pi05_fb_onpolicy",
            model=_fb_model(rep_head_dropout=0.1, fb_target_ema=0.99),
            freeze_filter=_fb_model().get_freeze_filter(),
            data=_rollout_data(),
            batch_size=32,
            lr_schedule=_schedule(),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
            num_train_steps=8_000,
            save_interval=1_000,
            keep_period=1_000,
        ),
        TrainConfig(
            # Single-variable companion to pi05_fb_onpolicy: proprioception concatenated onto both
            # heads' pooled vectors. Done at the heads, NOT via discrete_state_input, so the frozen
            # backbone's inputs -- and hence the base policy's actions -- are untouched.
            name="pi05_fb_onpolicy_proprio",
            model=_fb_model(rep_head_dropout=0.1, fb_target_ema=0.99, include_proprio=True),
            freeze_filter=_fb_model(include_proprio=True).get_freeze_filter(),
            data=_rollout_data(),
            batch_size=32,
            lr_schedule=_schedule(),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
            num_train_steps=8_000,
            save_interval=1_000,
            keep_period=1_000,
        ),
    ]
