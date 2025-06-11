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
    means: jax.Array,
    cholesky_diag: jax.Array,
    cholesky_off_diag: jax.Array,
    N: int,
    key: jax.Array,
) -> jax.Array:
    """Samples N points from a multivariate Gaussian distribution defined by Cholesky factors."""
    D = means.shape[0]

    # Create cholesky matrix L
    cholesky_l = jnp.zeros((D, D))
    # Ensure diagonal elements are positive to maintain positive definiteness
    cholesky_diag = jnp.maximum(cholesky_diag, 1e-6)
    # Set the diagonal and lower triangular elements
    cholesky_l = cholesky_l.at[jnp.diag_indices(D)].set(cholesky_diag)
    cholesky_l = cholesky_l.at[jnp.tril_indices(D, -1)].set(cholesky_off_diag)

    # Sample from a standard normal distribution (mean 0, identity covariance)
    standard_normal_samples = random.normal(key, shape=(N, D))
    # Transform the samples: X = mu + L @ z^T
    samples = means + jnp.dot(standard_normal_samples, cholesky_l.T)

    return samples


def _evaluate_bernstein_polynomials(
    samples: jax.Array, num_steps: int, degree: int
) -> jax.Array:
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
    action_sequences: jax.Array,
    initial_state: _env.PlanningAgentSimulatorState,
    rollout_env: _env.PlanningAgentEnvironment,
    num_steps: int,
    rollout_keys: jax.Array,
) -> _env.RolloutOutput:
    """Rolls out action sequences in the environment and returns outputs and metrics."""

    # Define simpler actor logic
    def init_fn(rng, state):
        return {"action_index": 0}

    def select_action(params, state, actor_state, rng):
        # Get current action directly
        action_index = actor_state["action_index"]
        action = params["action_sequence"][action_index]

        action_obj = datatypes.Action(data=action, valid=jnp.ones(1, dtype=bool))

        # Update state immutably
        new_actor_state = {"action_index": jnp.minimum(action_index + 1, num_steps - 1)}

        return agents.WaymaxActorOutput(
            actor_state=new_actor_state,
            action=action_obj,
            is_controlled=state.object_metadata.is_sdc,
        )

    sequence_actor = agents.actor_core_factory(init_fn, select_action)
    batched_actor_params = {"action_sequence": action_sequences}

    def single_rollout(rng_key, actor_params_single):
        return _env.rollout(
            scenario=initial_state,
            actor=sequence_actor,
            env=rollout_env,
            rng=rng_key,
            rollout_num_steps=num_steps,
            actor_params=actor_params_single,
        )

    # Vectorize the rollout function for all action sequences
    vmapped_rollout = jax.vmap(single_rollout, in_axes=(0, 0))
    return vmapped_rollout(rollout_keys, batched_actor_params)


def get_best_action(
    means: jax.Array,
    cholesky_diag: jax.Array,
    cholesky_off_diag: jax.Array,
    state: _env.PlanningAgentSimulatorState,
    current_action_sequence: jax.Array,
    rollout_env: _env.PlanningAgentEnvironment,
    N: int,
    horizon: float,
    random_key: jax.Array,
) -> jax.Array:
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

    # Find the sequence with highest reward
    rewards = rollout_outputs.reward
    total_reward_per_rollout = jnp.sum(rewards, axis=-1)

    # 4. Find the best action sequence
    best_index = jnp.argmax(total_reward_per_rollout)
    best_action_sequence = action_sequences[best_index]

    return best_action_sequence


jitted_get_best_action = jax.jit(
    get_best_action, static_argnames=["N", "rollout_env", "horizon"]
)
