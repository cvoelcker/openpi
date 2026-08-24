import dataclasses
import functools
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.rl_data_loader as _rl_data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights.

    Params present in params_shape but absent from the checkpoint (e.g. newly added
    phi/psi rep heads) are silently skipped — the model keeps its freshly-initialized values.
    """
    loaded_params = loader.load(params_shape)

    # Fill any newly-added params that the checkpoint doesn't know about with their
    # ShapeDtypeStruct placeholder so the tree structure matches for validation.
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    flat_shape = traverse_util.flatten_dict(params_shape)
    for k, v in flat_shape.items():
        if k not in flat_loaded:
            flat_loaded[k] = v
    loaded_params = traverse_util.unflatten_dict(flat_loaded)

    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove ShapeDtypeStruct placeholders — only actually-loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _as_batch_dict(batch) -> dict[str, Any]:
    """Normalize a data-loader batch to the model's batch-dict convention.

    The standard data loader yields (Observation, Actions) tuples; the goal-conditioned
    loader already yields dicts (observation, actions, future_observation, ...).
    """
    if isinstance(batch, dict):
        return batch
    observation, actions = batch
    return {"observation": observation, "actions": actions}


@at.typecheck
def _check_freeze_filter_matches_model(config: _config.TrainConfig) -> None:
    """Fail loudly when config.freeze_filter disagrees with what config.model implies.

    Configs must build freeze_filter from a duplicate model-config literal, since the `model`
    field is not referenceable from inside the same constructor call. A `--model.*` override that
    affects freezing then desyncs the two, and freeze_filter is tyro.conf.Suppress so it cannot be
    overridden to match. The failure is silent: the optimizer just never updates those params.
    """
    implied = getattr(config.model, "get_freeze_filter", None)
    if implied is None:
        return
    abstract = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
    frozen_configured = set(nnx.state(abstract, config.freeze_filter).flat_state())
    frozen_implied = set(nnx.state(abstract, implied()).flat_state())
    if frozen_configured != frozen_implied:
        only_cfg, only_impl = len(frozen_configured - frozen_implied), len(frozen_implied - frozen_configured)
        raise ValueError(
            f"freeze_filter disagrees with what config.model implies: {len(frozen_configured)} params frozen "
            f"by the configured filter vs {len(frozen_implied)} implied by the model "
            f"({only_cfg} only-configured, {only_impl} only-implied).\n"
            "This happens when a --model.* override changes freezing (action_loss_coeff, "
            "rep_backbone_grad_scale, lora variants) but freeze_filter -- which is "
            "tyro.conf.Suppress and cannot be overridden -- still reflects the config's hardcoded "
            "duplicate. The run would silently train the wrong parameter set.\n"
            "Fix: add a named config in config.py whose freeze_filter literal matches the model."
        )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    _check_freeze_filter_matches_model(config)
    tx = _optimizer.create_optimizer(
        config.optimizer,
        config.lr_schedule,
        weight_decay_mask=_optimizer.kernel_decay_mask(config.weight_decay_exclude_regex),
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        # Lagging target networks (e.g. Pi0SP's EMA psi) must start equal to their online
        # counterpart, including any weights the checkpoint just loaded into it.
        if (sync_targets := getattr(model, "sync_target_networks", None)) is not None:
            sync_targets()

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: dict,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng: at.KeyArrayLike, batch: dict):
        chunked_loss, log_dict = model.compute_loss(rng, batch, train=True)
        return jnp.mean(chunked_loss), log_dict

    train_rng = jax.random.fold_in(rng, state.step)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    ((loss, log_dict), grads) = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, batch)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    # EMA step for lagging target networks (e.g. Pi0SP's psi): they are excluded from
    # trainable_filter, so the optimizer never touches them and this mutation is what
    # carries into new_params below.
    if (update_targets := getattr(model, "update_target_networks", None)) is not None:
        update_targets()
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **{k: jnp.mean(v) for k, v in log_dict.items()},
    }
    return new_state, info


@at.typecheck
def val_step(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: dict,
) -> dict[str, at.Array]:
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()

    # The model reads the keys it needs from the batch (e.g. the CRL ranking probe picks up
    # the frame-index keys at val time) and ignores the rest.
    chunked_loss, log_dict = model.compute_loss(rng, batch, train=False)
    return {"val/loss": jnp.mean(chunked_loss), **{f"val/{k}": jnp.mean(v) for k, v in log_dict.items()}}


def main(config: _config.TrainConfig):
    init_logging()

    # Only initialize JAX distributed for genuine multi-node runs. Upstream calls this
    # unconditionally, but a single-node launch (no srun) has no SLURM_STEP_NODELIST and
    # crashes here; guarding on SLURM_NNODES>1 keeps single-node runs on all local GPUs.
    if int(os.environ.get("SLURM_NNODES", "1")) > 1:
        jax.distributed.initialize()

    is_main_process = jax.process_index() == 0

    if config.unique_run_id and jax.process_count() > 1:
        suffix_bytes = np.array(list(config._run_id_suffix.encode()), dtype=np.uint8)
        suffix_bytes = jax.experimental.multihost_utils.broadcast_one_to_all(suffix_bytes)
        synced_suffix = bytes(np.array(suffix_bytes, dtype=np.uint8)).decode()
        if config._run_id_suffix != synced_suffix:
            logging.info(f"Synced run_id_suffix: {config._run_id_suffix} -> {synced_suffix}")
            object.__setattr__(config, "_run_id_suffix", synced_suffix)

    logging.info(
        f"Running on: {platform.node()}, process {jax.process_index()}/{jax.process_count()}, "
        f"{jax.local_device_count()} local / {jax.device_count()} total devices"
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and is_main_process)

    # Goal-conditioned models (Pi0CRL, Pi0SF) train on HER-style dict batches from
    # rl_data_loader; everything else uses the standard supervised loader.
    if config.model.requires_goal_data:
        train_loader, val_loader = _rl_data_loader.create_train_val_goal_conditioned_data_loaders(
            config,
            sharding=data_sharding,
        )
    else:
        train_loader, val_loader = _data_loader.create_train_val_data_loaders(
            config,
            sharding=data_sharding,
        )
    data_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader is not None else None
    logging.info("About to fetch first batch from data loader...")
    t0 = time.time()
    batch = _as_batch_dict(next(data_iter))
    t1 = time.time()
    logging.info("Fetched first batch in %.3f s", t1 - t0)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pval_step = (
        jax.jit(
            val_step,
            in_shardings=(train_state_sharding, replicated_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        if val_loader is not None
        else None
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not is_main_process,
    )

    infos = []
    # Best-val checkpoint tracking (no-op unless config.save_best_val). Not persisted across
    # a resume: after resuming, `best_val` restarts at inf, so the first val pass will look
    # like an improvement and write one extra checkpoint.
    best_val = float("inf")
    saved_this_step = -1
    logged_missing_metric = False
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if is_main_process:
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
        batch = _as_batch_dict(next(data_iter))

        if pval_step is not None and val_iter is not None and step % config.val_interval == 0:
            val_rng = jax.random.fold_in(train_rng, step)
            val_infos = []
            for _ in range(config.val_batches):
                val_batch = _as_batch_dict(next(val_iter))
                with sharding.set_mesh(mesh):
                    vi = pval_step(train_state, val_rng, val_batch)
                val_infos.append(vi)
            stacked_val = common_utils.stack_forest(val_infos)
            reduced_val = jax.device_get(jax.tree.map(jnp.mean, stacked_val))
            if is_main_process:
                val_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val.items())
                pbar.write(f"Step {step} [val]: {val_str}")
                wandb.log(reduced_val, step=step)
            # Best-val checkpointing: training always runs the full num_train_steps and saves
            # on a fixed interval, so without this the kept checkpoints have no relationship
            # to generalization — at ~95 epochs over LIBERO the last one is rarely the best.
            if config.save_best_val and step > start_step:
                metric = reduced_val.get(config.best_val_metric)
                if metric is None:
                    if step == start_step or not logged_missing_metric:
                        logging.warning(
                            "save_best_val is on but %r is not in the val metrics (have: %s); "
                            "no best checkpoint will be written",
                            config.best_val_metric,
                            sorted(reduced_val),
                        )
                        logged_missing_metric = True
                elif float(metric) < best_val:
                    best_val = float(metric)
                    if is_main_process:
                        pbar.write(f"Step {step}: new best {config.best_val_metric}={best_val:.4f}, saving")
                    _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)
                    saved_this_step = step

        # `step != saved_this_step` skips a duplicate write when the best-val branch above
        # already saved this exact step.
        if ((step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1) and (
            step != saved_this_step
        ):
            _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
