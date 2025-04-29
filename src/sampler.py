from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from jax.lax import lgamma
from waymax import agents, datatypes
from waymax import env as _env


def binom_jax(n, k):
    """Computes the binomial coefficient using lgamma for stability."""
    # Ensure inputs are appropriate for lgamma (e.g., non-negative)
    # Convert to float for lgamma compatibility
    n_float = n + 1.0
    k_float = k + 1.0
    n_minus_k_float = n - k + 1.0
    # Add small epsilon to avoid log(0) issues if needed, though lgamma handles positive integers.
    # Check for k < 0 or k > n if necessary.
    return jnp.exp(lgamma(n_float) - lgamma(k_float) - lgamma(n_minus_k_float))


def _sample_distribution(
    means: jnp.ndarray,
    cholesky_diag: jnp.ndarray,
    cholesky_off_diag: jnp.ndarray,
    N: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Samples N points from a multivariate Gaussian distribution defined by Cholesky factors."""
    D = means.shape[0]

    # Create cholesky matrix
    cholesky_l = jnp.zeros((D, D))
    cholesky_diag = jnp.maximum(cholesky_diag, 1e-6)  # Ensure positive definiteness
    cholesky_l = cholesky_l.at[jnp.diag_indices(D)].set(cholesky_diag)
    cholesky_l = cholesky_l.at[jnp.tril_indices(D, -1)].set(cholesky_off_diag)

    # Create the covariance matrix using the Cholesky factorization
    covariance_matrix = jnp.dot(cholesky_l, cholesky_l.T)
    covariance_matrix = covariance_matrix + jnp.eye(D) * 1e-6  # Add jitter

    # Sample from the distribution
    samples = random.multivariate_normal(
        key, means, covariance_matrix, shape=(N,), method="svd"
    )
    return samples


def _evaluate_bernstein_polynomials(
    samples: jnp.ndarray, num_steps: int, degree: int
) -> jnp.ndarray:
    """Evaluates Bernstein polynomials based on sampled coefficients."""
    N, D = samples.shape
    num_params_per_poly = D // 2

    # Split the samples into two equal parts for two polynomials
    poly1_samples = samples[:, :num_params_per_poly]
    poly2_samples = samples[:, num_params_per_poly:]

    # Transform coefficients using tanh
    transformed_poly1_coeffs = jnp.tanh(poly1_samples)
    transformed_poly2_coeffs = jnp.tanh(poly2_samples)

    # Generate normalized time steps [0, 1]
    t_norm = jnp.linspace(0, 1, num_steps)

    # Define Bernstein polynomial evaluation function
    def bernstein_poly_eval_norm(coeffs, t, n):
        i = jnp.arange(n + 1)
        t_pow_i = jnp.where(i == 0, 1.0, t**i)
        one_minus_t_pow_n_minus_i = jnp.where(n - i == 0, 1.0, (1 - t) ** (n - i))
        basis = binom_jax(n, i) * t_pow_i * one_minus_t_pow_n_minus_i
        return jnp.sum(coeffs * basis, axis=-1)

    # Vectorize the evaluation function
    eval_single_poly_over_time = jax.vmap(
        lambda c, t: bernstein_poly_eval_norm(c, t, degree), in_axes=(None, 0)
    )
    eval_all_polys_over_time = jax.vmap(eval_single_poly_over_time, in_axes=(0, None))

    # Evaluate both sets of polynomials
    poly1_values = eval_all_polys_over_time(transformed_poly1_coeffs, t_norm)
    poly2_values = eval_all_polys_over_time(transformed_poly2_coeffs, t_norm)

    # Stack the results to form action sequences
    action_sequences = jnp.stack([poly1_values, poly2_values], axis=-1)
    return action_sequences


def _rollout_action_sequences(
    action_sequences: jnp.ndarray,
    initial_state: _env.PlanningAgentSimulatorState,
    rollout_env: _env.PlanningAgentEnvironment,
    num_steps: int,
    rollout_keys: jnp.ndarray,
) -> _env.RolloutOutput:
    """Rolls out action sequences in the environment and returns outputs and metrics."""

    # Define actor logic
    init = lambda rng, state: {"action_index": 0}

    def select_action(
        params: Any, state: datatypes.SimulatorState, actor_state: Any, rng: jax.Array
    ):
        action_index = actor_state["action_index"]
        action_sequence = params["action_sequence"]
        current_action = jax.lax.dynamic_slice_in_dim(
            action_sequence, action_index, 1, axis=0
        )
        action = jnp.squeeze(current_action, axis=0)
        action_obj = datatypes.Action(
            data=jnp.array([action[0], action[1]]), valid=jnp.array([True])
        )
        actor_state["action_index"] += 1
        return agents.WaymaxActorOutput(
            actor_state=actor_state,  # type: ignore
            action=action_obj,  # type: ignore
            is_controlled=state.object_metadata.is_sdc,  # type: ignore
        )

    sequence_actor = agents.actor_core_factory(init, select_action)
    batched_actor_params = {"action_sequence": action_sequences}

    # Define and vmap the rollout function
    def single_rollout(rng_key, actor_params_single):
        return _env.rollout(
            scenario=initial_state,
            actor=sequence_actor,
            env=rollout_env,
            rng=rng_key,
            rollout_num_steps=num_steps,
            actor_params=actor_params_single,
        )

    vmapped_rollout = jax.vmap(single_rollout, in_axes=(0, 0))
    rollout_outputs = vmapped_rollout(rollout_keys, batched_actor_params)

    return rollout_outputs


def get_best_action(
    means: jnp.ndarray,
    cholesky_diag: jnp.ndarray,
    cholesky_off_diag: jnp.ndarray,
    state: _env.PlanningAgentSimulatorState,
    current_action_sequence: jnp.ndarray,
    rollout_env: _env.PlanningAgentEnvironment,
    N: int,
    horizon: float,
    random_key: jax.Array,
) -> jnp.ndarray:
    """Computes the best action sequence by sampling, evaluating polynomials, and rolling out.

    Args:
        means: Mean vector of the Gaussian distribution.
        cholesky_diag: Diagonal elements of the Cholesky factor.
        cholesky_off_diag: Off-diagonal elements of the Cholesky factor (lower triangle).
        state: Current state.
        current_action_sequence: Current action sequence to be evaluated.
        rollout_env: Environment to roll out the actions.
        N: Number of action sequences to sample and evaluate.
        horizon: Planning horizon in seconds.
        random_key: Random key for sampling.

    Returns:
        The best action sequence based on minimizing overlap during rollouts.
    """
    D = means.shape[0]
    num_params_per_poly = D // 2
    degree = num_params_per_poly - 1

    DT = 0.1  # Timestep duration in seconds
    num_steps = int(round(horizon / DT))

    # 1. Sample from the distribution
    key, subkey = random.split(random_key)
    samples = _sample_distribution(means, cholesky_diag, cholesky_off_diag, N, subkey)

    # 2. Create and evaluate Bernstein polynomials to get action sequences
    action_sequences = _evaluate_bernstein_polynomials(samples, num_steps, degree)
    # Add the current action sequence to the batch
    action_sequences = jnp.concatenate(
        [action_sequences, jnp.expand_dims(current_action_sequence, axis=0)], axis=0
    )

    # 3. Rollout action sequences and evaluate
    rollout_keys = random.split(key, len(action_sequences))
    rollout_outputs = _rollout_action_sequences(
        action_sequences, state, rollout_env, num_steps, rollout_keys
    )
    # Extract and sum the overlap metric
    overlap_metrics = rollout_outputs.metrics["overlap"]
    total_overlap_per_rollout = jnp.sum(overlap_metrics.value, axis=1)

    progress_metrics = rollout_outputs.metrics["sdc_progression"]
    total_progress_per_rollout = jnp.sum(progress_metrics.value, axis=1)

    # Find the sequence with highest reward
    # rewards = rollout_outputs.reward
    # total_reward_per_rollout = jnp.sum(rewards, axis=-1)

    # 4. Find the best action sequence
    best_index = jnp.argmax(total_progress_per_rollout - total_overlap_per_rollout)
    best_action_sequence = action_sequences[best_index]

    return best_action_sequence


jitted_get_best_action = jax.jit(get_best_action, static_argnames=["N", "rollout_env"])
