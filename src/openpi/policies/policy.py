from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def set_sample_kwarg(self, key: str, value) -> None:
        """Set one keyword forwarded to the model's sample_actions* call, so callers can adjust
        sampling on an already-constructed policy without reaching into `_sample_kwargs`."""
        self._sample_kwargs = {**self._sample_kwargs, key: value}

    def reset_rng(self, seed: int) -> None:
        """Re-seed the sampling RNG, so a closed-loop rollout's action noise does not depend on
        how long every preceding episode happened to run.

        Without it, four byte-identical eval runs scored 0.510/0.460/0.470/0.490 and disagreed on
        14.5% of episodes -- more than the treatment being measured. No-op for PyTorch policies.
        """
        if not self._is_pytorch_model:
            self._rng = jax.random.key(seed)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batch(self, obs_list: Sequence[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        """Batched counterpart to `infer`: the same per-example transform pipeline, but one
        `sample_actions` call with a real leading batch dim instead of `len(obs_list)` calls.

        `noise`, if given, must already carry that leading batch dim -- unlike `infer()`'s, which
        accepts an unbatched array and adds it.
        """
        n = len(obs_list)
        if n == 0:
            return []

        per_example_inputs = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]
        batched_inputs = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *per_example_inputs)

        if not self._is_pytorch_model:
            batched_inputs = jax.tree.map(jnp.asarray, batched_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            batched_inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), batched_inputs
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(batched_inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time_ms = (time.monotonic() - start_time) * 1000

        outputs_batched = {"state": batched_inputs["state"], "actions": actions}
        if self._is_pytorch_model:
            outputs_batched = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs_batched)
        else:
            outputs_batched = jax.tree.map(np.asarray, outputs_batched)

        results = []
        for i in range(n):
            per_example_out = self._output_transform(jax.tree.map(lambda x, i=i: x[i], outputs_batched))
            per_example_out["policy_timing"] = {"infer_ms": model_time_ms, "batch_size": n}
            results.append(per_example_out)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
