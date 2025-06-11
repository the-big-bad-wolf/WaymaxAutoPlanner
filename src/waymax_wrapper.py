# Description: Wrapper for Waymax environment to be compatible with skrl
import dataclasses
import glob
import os
import time
from typing import Any, Iterator, Tuple, Union, override

import cv2
import gymnasium
import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import skrl.envs.wrappers.jax as skrl_wrappers
from dm_env import specs
from jax import jit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from waymax import agents
from waymax import config as _config
from waymax import datatypes, dynamics
from waymax import env as _env
from waymax import visualization

from sampler import jitted_get_best_action
from waymax_modified import WaymaxEnv


def merged_step(
    env: _env.PlanningAgentEnvironment,
    state: _env.PlanningAgentSimulatorState,
    actions: jax.Array,
):
    action = datatypes.Action(data=actions.flatten(), valid=jnp.ones((1,), dtype=bool))
    new_state = env.step(state, action)
    reward = env.reward(new_state, action).reshape(1, -1)
    observation = env.observe(new_state).reshape(1, -1)
    terminated = env.termination(new_state).reshape(1, -1)
    truncated = env.truncation(new_state).reshape(1, -1)
    metrics = env.metrics(new_state)
    return new_state, observation, reward, terminated, truncated, metrics


merged_step = jit(merged_step, static_argnums=(0,))


def merged_multistep(
    env: _env.PlanningAgentEnvironment,
    state: _env.PlanningAgentSimulatorState,
    actions: jax.Array,
    num_steps: int,
):
    """Performs a multi-step rollout using JAX for loop with environment steps."""

    def step_body(i, carry):
        current_state, all_states, all_rewards = carry

        # Get action for current step
        action_data = actions[i]
        action = datatypes.Action(data=action_data, valid=jnp.ones(1, dtype=bool))

        # Step the environment
        new_state = env.step(current_state, action)
        reward = env.reward(new_state, action)

        # Update arrays using jax.tree_map for states
        all_states = jax.tree_map(lambda x, y: x.at[i].set(y), all_states, new_state)
        all_rewards = all_rewards.at[i].set(reward)

        return new_state, all_states, all_rewards

    # Initialize storage arrays using the current state structure and reward spec
    all_states = jax.tree_map(
        lambda x: jnp.zeros((num_steps,) + x.shape, dtype=x.dtype), state
    )

    # Get reward shape from environment spec
    reward_spec = env.reward_spec()
    all_rewards = jnp.zeros((num_steps,) + reward_spec.shape, dtype=reward_spec.dtype)

    # Run the for loop
    final_state, all_states, all_rewards = jax.lax.fori_loop(
        0, num_steps, step_body, (state, all_states, all_rewards)
    )

    # Get final observation and reshape
    observation = env.observe(final_state).reshape(1, -1)

    # Calculate metrics from all states
    all_metrics = jax.vmap(env.metrics)(all_states)

    # Detect overlap/offroad events across all steps
    overlap_detected = jnp.any(all_metrics["overlap"].value > 0).astype(jnp.int32)
    offroad_detected = jnp.any(all_metrics["offroad"].value > 0).astype(jnp.int32)

    # Get final progression
    rollout_progression = all_metrics["sdc_progression"].value[-1]

    # Sum all rewards and reshape
    total_reward = jnp.sum(all_rewards).reshape(1, -1)

    # Get termination and truncation for final state
    terminated = env.termination(final_state).reshape(1, -1)
    truncated = env.truncation(final_state).reshape(1, -1)

    return (
        final_state,
        observation,
        total_reward,
        terminated,
        truncated,
        overlap_detected,
        offroad_detected,
        rollout_progression,
        all_states,
        all_rewards,
    )


merged_multistep = jit(merged_multistep, static_argnums=(0, 3))


def merged_reset(
    env: _env.PlanningAgentEnvironment, scenario: datatypes.SimulatorState
):
    state = env.reset(scenario)
    observation = env.observe(state).reshape(1, -1)
    return state, observation


merged_reset = jit(merged_reset, static_argnums=(0,))


@jit
def jerk_reward(actions: jax.Array, prev_actions: jax.Array) -> jax.Array:
    """Calculate jerk reward based on the difference between the current and previous actions."""
    accel_jerk = jnp.abs(actions[0] - prev_actions[0]) / 2
    steering_jerk = jnp.abs(actions[1] - prev_actions[1]) / 2
    return -0.1 * accel_jerk - 0.5 * steering_jerk


@jit
def _update_episode_metrics(
    current_had_overlap: jnp.ndarray,
    current_had_offroad: jnp.ndarray,
    overlap_detected: jnp.ndarray,
    offroad_detected: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update episode metrics in a JIT-compatible way"""
    # Use maximum to avoid conditionals - if either is 1, result is 1
    new_had_overlap = jnp.maximum(current_had_overlap, overlap_detected)
    new_had_offroad = jnp.maximum(current_had_offroad, offroad_detected)
    return new_had_overlap, new_had_offroad


@jit
def _detect_events_from_metrics(metrics):
    """Detect overlap and offroad events from metrics in a JIT-compatible way"""
    # Extract the relevant metrics and convert to binary flags
    overlap_detected = jnp.any(metrics["overlap"].value > 0).astype(jnp.int32)
    offroad_detected = jnp.any(metrics["offroad"].value > 0).astype(jnp.int32)
    progression = metrics["sdc_progression"].value
    return overlap_detected, offroad_detected, progression


@jit
def _update_episode_counters(
    episode_total: jnp.ndarray,
    episodes_with_overlap: jnp.ndarray,
    episodes_with_offroad: jnp.ndarray,
    total_progression: jnp.ndarray,
    had_overlap: jnp.ndarray,
    had_offroad: jnp.ndarray,
    episode_progression: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update episode counters when an episode ends"""
    new_total = episode_total + 1
    new_overlap = episodes_with_overlap + had_overlap
    new_offroad = episodes_with_offroad + had_offroad
    new_total_progression = total_progression + episode_progression
    return new_total, new_overlap, new_offroad, new_total_progression


class WaymaxWrapper(skrl_wrappers.Wrapper):
    def __init__(
        self,
        env: _env.PlanningAgentEnvironment,
        scenario_loader: Iterator[datatypes.SimulatorState],
        action_space_type: str = "bicycle",
        save_dir: str | None = None,  # Add save directory parameter
        save_videos: bool = True,  # Option to disable video saving
    ):
        """
        Initialize the Waymax wrapper.

        Args:
            env: The Waymax planning agent environment.
            scenario_loader: Iterator providing simulator states.
            action_space_type: One of "bicycle", "bicycle_mpc", or "trajectory_sampling".
            save_dir: Directory to save videos and statistics. If None, uses default behavior.
            save_videos: Whether to generate and save videos (can be disabled for faster training).
        """
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._random_key = jax.random.key(int(time.time()))
        self._action_space_type = action_space_type
        self._prev_action = jnp.zeros((2,), dtype=np.float32)
        self._save_dir = save_dir
        self._save_videos = save_videos

        # Initialize video storage only if videos are enabled
        if self._save_videos:
            self._states = []
            self._rewards = []
            if action_space_type == "trajectory_sampling":
                self._action_sequences = []

        # Set up the appropriate step and reset methods based on action_space_type
        if action_space_type == "trajectory_sampling":
            self._step_impl = self._step_trajectory_sampling
            self._reset_impl = self._reset_trajectory_sampling
            self._setup_trajectory_sampling()
        elif action_space_type == "bicycle_mpc":
            self._get_MPC_action = None
            if action_space_type == "bicycle_mpc":
                from mpc import get_MPC_action

                self._get_MPC_action = get_MPC_action
            self._step_impl = self._step_bicycle_mpc
            self._reset_impl = self._reset_bicycle_mpc
        elif action_space_type == "bicycle":
            self._step_impl = self._step_bicycle_no_mpc
            self._reset_impl = self._reset_bicycle_no_mpc
        else:
            raise ValueError(
                f"Unknown action_space_type: {action_space_type}. "
                "Must be one of: 'bicycle', 'bicycle_mpc', 'trajectory_sampling'"
            )

        self._setup_metrics_tracking()

    def _setup_metrics_tracking(self):
        """Set up metrics tracking for episodes"""
        # Initialize episode counters as JAX arrays
        self._episodes_total = jnp.array(-1, dtype=jnp.int32)
        self._episodes_with_overlap = jnp.array(0, dtype=jnp.int32)
        self._episodes_with_offroad = jnp.array(0, dtype=jnp.int32)
        self._total_progression = jnp.array(
            0.0, dtype=jnp.float32
        )  # Total progression across all episodes
        # Current episode tracking flags
        self._current_episode_had_overlap = jnp.array(0, dtype=jnp.int32)
        self._current_episode_had_offroad = jnp.array(0, dtype=jnp.int32)
        self._current_episode_progression = jnp.array(
            0.0, dtype=jnp.float32
        )  # Current episode's progression

    def _setup_trajectory_sampling(self):
        """Setup for trajectory sampling action space"""
        self._nr_rollouts = 10  # Number of trajectories to sample
        self._horizon = 3  # Horizon in seconds
        self._DT = 0.1  # Time step duration
        # Configuration for the polynomial coefficients distribution
        num_polys = 2
        poly_degree = 3
        num_coeffs_per_poly = poly_degree + 1
        self._mean_dim = num_polys * num_coeffs_per_poly
        self._cholesky_diag_dim = self._mean_dim
        self._cholesky_offdiag_dim = self._mean_dim * (self._mean_dim - 1) // 2

        # Set up rollout environment
        constant_actor = agents.create_constant_speed_actor(
            dynamics_model=self._env._state_dynamics,
            is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
            speed=None,
        )
        metrics_config = _config.MetricsConfig(
            metrics_to_run=("sdc_progression", "offroad", "overlap")
        )
        reward_config = _config.LinearCombinationRewardConfig(
            rewards={"sdc_progression": 1.0, "offroad": -2.0, "overlap": -4.0},
        )
        env_config = dataclasses.replace(
            _config.EnvironmentConfig(),
            metrics=metrics_config,
            rewards=reward_config,
            compute_reward=True,
        )
        self._rollout_env = WaymaxEnv(
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            config=env_config,
            sim_agent_actors=[constant_actor],
            sim_agent_params=[{}],
        )
        self._replan_interval = 2  # seconds

    @override
    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        (
            self._episodes_total,
            self._episodes_with_overlap,
            self._episodes_with_offroad,
            self._total_progression,
        ) = _update_episode_counters(
            self._episodes_total,
            self._episodes_with_overlap,
            self._episodes_with_offroad,
            self._total_progression,
            self._current_episode_had_overlap,
            self._current_episode_had_offroad,
            self._current_episode_progression,
        )

        # Reset episode tracking flags
        self._current_episode_had_overlap = jnp.zeros_like(
            self._current_episode_had_overlap
        )
        self._current_episode_had_offroad = jnp.zeros_like(
            self._current_episode_had_offroad
        )
        self._current_episode_progression = jnp.zeros_like(
            self._current_episode_progression
        )

        return self._reset_impl()

    def _reset_bicycle_no_mpc(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset implementation for bicycle action space without MPC"""
        scenario = next(self._scenario_loader)
        self._current_state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)

        return observation, {}

    def _reset_bicycle_mpc(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset implementation for bicycle action space with MPC"""
        scenario = next(self._scenario_loader)
        self._current_state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)

        # Get MPC action and append to observation
        self._MPC_action = self._get_MPC_action(self._current_state)  # type: ignore
        mpc_action_array = np.array(self._MPC_action).reshape(1, 2)
        observation = np.concatenate([observation, mpc_action_array], axis=1)

        return observation, {}

    def _reset_trajectory_sampling(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset implementation for trajectory sampling action space"""
        scenario = next(self._scenario_loader)
        self._current_state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)

        self._current_action_plan = jnp.zeros(
            (int(round(self._horizon / self._DT)), 2), dtype=jnp.float32
        )

        return observation, {}

    @override
    def step(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Perform a step in the environment"""
        actions = actions.flatten()
        return self._step_impl(actions)

    def _step_bicycle_no_mpc(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Step implementation for bicycle action space without MPC"""
        accel, steering = actions

        # Reshape for the environment step
        action_array = jnp.array([accel, steering], dtype=np.float32)
        (
            self._current_state,
            observation,
            reward,
            terminated,
            truncated,
            metrics,
        ) = merged_step(self._env, self._current_state, action_array)
        reward += jerk_reward(action_array, self._prev_action)

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)

        # Detect events from metrics
        overlap_detected, offroad_detected, progression = _detect_events_from_metrics(
            metrics
        )

        # Update episode tracking flags
        self._current_episode_had_overlap, self._current_episode_had_offroad = (
            _update_episode_metrics(
                self._current_episode_had_overlap,
                self._current_episode_had_offroad,
                overlap_detected,
                offroad_detected,
            )
        )

        self._current_episode_progression = progression

        self._prev_action = action_array
        self._current_reward: float = reward[0, 0]

        return observation, reward, terminated, truncated, {}

    def _step_bicycle_mpc(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Step implementation for bicycle action space with MPC"""
        accel, steering = actions
        mpc_accel, mpc_steering = self._MPC_action
        accel = accel + mpc_accel
        steering = steering + mpc_steering

        # Reshape for the environment step
        action_array = jnp.array([accel, steering], dtype=np.float32)
        (
            self._current_state,
            observation,
            reward,
            terminated,
            truncated,
            metrics,
        ) = merged_step(self._env, self._current_state, action_array)
        reward += jerk_reward(action_array, self._prev_action)

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)

        # Get MPC action for next step and append to observation
        self._MPC_action = self._get_MPC_action(self._current_state)  # type: ignore
        mpc_action_array = np.array(self._MPC_action).reshape(1, 2)
        observation = np.concatenate([observation, mpc_action_array], axis=1)

        # Detect events from metrics
        overlap_detected, offroad_detected, progression = _detect_events_from_metrics(
            metrics
        )

        # Update episode tracking flags
        self._current_episode_had_overlap, self._current_episode_had_offroad = (
            _update_episode_metrics(
                self._current_episode_had_overlap,
                self._current_episode_had_offroad,
                overlap_detected,
                offroad_detected,
            )
        )

        self._current_episode_progression = progression

        self._prev_action = action_array
        self._current_reward: float = reward[0, 0]

        return observation, reward, terminated, truncated, {}

    def _step_trajectory_sampling(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Step implementation for trajectory sampling action space - using multi-step rollout"""
        # Extract the mean and Cholesky parameters from the action vector
        means = jnp.array(actions[: self._mean_dim])
        cholesky_diag = jnp.array(
            actions[self._mean_dim : self._mean_dim + self._cholesky_diag_dim]
        )
        cholesky_offdiag = jnp.array(
            actions[
                self._mean_dim
                + self._cholesky_diag_dim : self._mean_dim
                + self._cholesky_diag_dim
                + self._cholesky_offdiag_dim
            ]
        )

        # Get the best action sequence from the Gaussian polynomial distribution
        self._random_key, subkey = jax.random.split(self._random_key)
        action_sequence = jitted_get_best_action(
            means,
            cholesky_diag,
            cholesky_offdiag,
            self._current_state,
            self._current_action_plan,
            self._rollout_env,
            self._nr_rollouts,
            self._horizon,
            subkey,
        )

        # Use the full action sequence for multi-step rollout
        num_steps = int(self._replan_interval / self._DT)

        (
            self._current_state,
            observation,
            reward,
            terminated,
            truncated,
            overlap_detected,
            offroad_detected,
            progression,
            all_states,
            all_rewards,
        ) = merged_multistep(self._env, self._current_state, action_sequence, num_steps)

        """
        if self._save_videos:
            # Convert JAX arrays to numpy and store each state
            for i in range(
                all_states.sim_trajectory.x.shape[0]
            ):  # Iterate over timesteps
                # Extract state at timestep i
                state_i = jax.tree_map(lambda x: x[i], all_states)
                reward_i = float(all_rewards[i])

                self._states.append(state_i)
                self._rewards.append(reward_i)
                self._action_sequences.append(np.array(action_sequence))
        """

        # Update episode tracking flags
        self._current_episode_had_overlap, self._current_episode_had_offroad = (
            _update_episode_metrics(
                self._current_episode_had_overlap,
                self._current_episode_had_offroad,
                overlap_detected,
                offroad_detected,
            )
        )

        # Store the current progression
        self._current_episode_progression = progression

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)
        self._current_reward = reward[0, 0]

        # Reset action plan for next iteration (since we executed the full sequence)
        self._current_action_plan = jnp.zeros(
            (int(round(self._horizon / self._DT)), 2), dtype=jnp.float32
        )

        return observation, reward, terminated, truncated, {}

    @override
    def render(self, *args, **kwargs) -> Any:
        """Store the state for video generation on close"""
        if not self._save_videos:
            return

        # Store the current state and reward for rendering
        self._states.append(self._current_state)
        self._rewards.append(self._current_reward)

        # Store the current action sequence for visualization if needed
        if self._action_space_type == "trajectory_sampling":
            self._action_sequences.append(self._current_action_plan.copy())

    @override
    def close(self) -> None:
        """Close the environment and save logs to specified directory"""

        # Use provided save directory or fall back to default behavior
        if self._save_dir is not None:
            save_directory = self._save_dir
            os.makedirs(save_directory, exist_ok=True)
            print(f"Saving logs to: {save_directory}")
        else:
            # Fall back to original behavior
            save_directory = self._get_default_save_directory()

        # Always save episode statistics
        self._save_episode_statistics(save_directory)

        # Save videos only if enabled and we have data
        if self._save_videos and hasattr(self, "_states") and self._states:
            self._save_videos_to_directory(save_directory)

        # Clean up
        self._cleanup_video_data()

    def _get_default_save_directory(self) -> str:
        """Get default save directory (original behavior)"""
        runs_dir = "runs/"
        os.makedirs(runs_dir, exist_ok=True)

        run_folders = glob.glob(os.path.join(runs_dir, "*"))
        eval_folders = glob.glob(os.path.join(runs_dir, "*/eval_logs"))

        if eval_folders:
            newest_eval_folder = max(eval_folders, key=os.path.getctime)
            return newest_eval_folder
        else:
            if not run_folders:
                newest_folder = os.path.join(runs_dir, "default")
                os.makedirs(newest_folder, exist_ok=True)
            else:
                newest_folder = max(run_folders, key=os.path.getctime)
            return newest_folder

    def _save_episode_statistics(self, save_directory: str) -> None:
        """Save episode statistics to file"""
        stats = self.get_episode_statistics()
        stats_path = os.path.join(save_directory, "episode_statistics.txt")

        with open(stats_path, "w") as f:
            f.write("Waymax Episode Statistics\n")
            f.write("=======================\n\n")
            f.write(f"Total Episodes: {stats['total_episodes']}\n")
            f.write(
                f"Episodes with Overlap: {stats['episodes_with_overlap']} ({stats['overlap_percentage']:.2f}%)\n"
            )
            f.write(
                f"Episodes with Offroad: {stats['episodes_with_offroad']} ({stats['offroad_percentage']:.2f}%)\n"
            )
            f.write(f"Average Progression: {stats['average_progression']:.2f}%\n")

        print(f"Episode statistics saved to '{stats_path}'")

    def _save_videos_to_directory(self, save_directory: str) -> None:
        """Save waymax and action sequence videos"""
        waymax_video_path = os.path.join(save_directory, "waymax.mp4")
        action_seq_video_path = os.path.join(save_directory, "action_sequences.mp4")

        # Generate waymax simulation video
        imgs = []
        for i in range(len(self._states)):
            img = visualization.plot_simulator_state(
                self._states[i], use_log_traj=False
            )

            # Add reward overlay
            reward_text = f"Reward: {self._rewards[i]:.4f}"
            img_with_text = self._add_text_overlay(img, reward_text)
            imgs.append(img_with_text)

        mediapy.write_video(waymax_video_path, imgs, fps=10)
        print(f"Waymax video saved to '{waymax_video_path}'")

        # Generate action sequence video if available
        if hasattr(self, "_action_sequences") and len(self._action_sequences) > 0:
            self._save_action_sequence_video(action_seq_video_path)

    def _add_text_overlay(self, img: np.ndarray, text: str) -> np.ndarray:
        """Add text overlay to image"""
        img_with_text = img.copy()
        h, w = img_with_text.shape[:2]

        # Create semi-transparent overlay
        overlay = img_with_text.copy()
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width = text_size[0] + 10

        pt1 = (w - text_width - 10, h - 40)
        pt2 = (w - 10, h - 10)
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)

        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img_with_text, 1 - alpha, 0, img_with_text)

        # Add text
        cv2.putText(
            img_with_text,
            text,
            (w - text_width - 5, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return img_with_text

    def _cleanup_video_data(self) -> None:
        """Clean up stored video data"""
        if hasattr(self, "_states"):
            self._states.clear()
        if hasattr(self, "_rewards"):
            self._rewards.clear()
        if hasattr(self, "_action_sequences"):
            self._action_sequences.clear()

    @property
    def observation_space(self) -> gymnasium.Space:
        """The observation specs of this environment, without batch dimension."""
        observation_spec: specs.BoundedArray = self._env.observation_spec()

        if self._action_space_type == "bicycle_mpc":
            # Add MPC action to the observation space
            # Assuming the MPC action is a 2D vector (acceleration, steering)
            observation_spec = specs.BoundedArray(
                shape=(observation_spec.shape[0] + 2,),
                minimum=np.concatenate(
                    [observation_spec.minimum, np.array([-1.0, -1.0])]
                ),
                maximum=np.concatenate(
                    [observation_spec.maximum, np.array([1.0, 1.0])]
                ),
                dtype=observation_spec.dtype,
            )

        return gymnasium.spaces.Box(
            shape=observation_spec.shape,  # type: ignore
            low=observation_spec.minimum,
            high=observation_spec.maximum,
            dtype=observation_spec.dtype,  # type: ignore
        )

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        if self._action_space_type == "trajectory_sampling":
            # Total dimension of the action space vector
            total_dim = (
                self._mean_dim + self._cholesky_diag_dim + self._cholesky_offdiag_dim
            )

            # Define bounds for the action space components.
            mean_min = np.full((self._mean_dim,), -1.0, dtype=np.float32)
            mean_max = np.full((self._mean_dim,), 1.0, dtype=np.float32)

            cholesky_diag_min = np.full(
                (self._cholesky_diag_dim,), 1e-6, dtype=np.float32
            )
            cholesky_diag_max = np.full(
                (self._cholesky_diag_dim,), np.inf, dtype=np.float32
            )

            cholesky_offdiag_min = np.full(
                (self._cholesky_offdiag_dim,), -np.inf, dtype=np.float32
            )
            cholesky_offdiag_max = np.full(
                (self._cholesky_offdiag_dim,), np.inf, dtype=np.float32
            )

            min_bounds = np.concatenate(
                [mean_min, cholesky_diag_min, cholesky_offdiag_min], axis=0
            )
            max_bounds = np.concatenate(
                [mean_max, cholesky_diag_max, cholesky_offdiag_max], axis=0
            )

            return gymnasium.spaces.Box(
                low=min_bounds,
                high=max_bounds,
                shape=(total_dim,),
                dtype=np.float32,
            )
        elif self._action_space_type in ["bicycle", "bicycle_mpc"]:
            # Original simple action space (e.g., acceleration, steering)
            action_spec: specs.BoundedArray = self._env.action_spec().data  # type: ignore
            if self._action_space_type == "bicycle_mpc":
                action_spec = specs.BoundedArray(
                    shape=(action_spec.shape[0],),
                    minimum=np.array([-2.0, -2.0]),
                    maximum=np.array([2.0, 2.0]),
                    dtype=action_spec.dtype,
                )
            return gymnasium.spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=(2,),
                dtype=action_spec.dtype,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown action_space_type: {self._action_space_type}")

    def get_episode_statistics(self):
        """Returns the current episode statistics using JAX arrays"""
        episodes_total = np.array(self._episodes_total)
        episodes_with_overlap = np.array(self._episodes_with_overlap)
        episodes_with_offroad = np.array(self._episodes_with_offroad)
        total_progression = np.array(self._total_progression)

        # Avoid division by zero
        safe_denom = np.maximum(1, episodes_total)

        overlap_percentage = (episodes_with_overlap / safe_denom) * 100.0
        offroad_percentage = (episodes_with_offroad / safe_denom) * 100.0
        average_progression = (total_progression / safe_denom) * 100

        return {
            "total_episodes": episodes_total,
            "episodes_with_overlap": episodes_with_overlap,
            "episodes_with_offroad": episodes_with_offroad,
            "overlap_percentage": overlap_percentage,
            "offroad_percentage": offroad_percentage,
            "average_progression": average_progression,
        }

    def _save_action_sequence_video(self, action_seq_video_path: str) -> None:
        """Generate and save action sequence visualization video"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            # Create figure for action sequence plots
            fig_width, fig_height = 12, 8

            imgs = []
            for i, action_seq in enumerate(self._action_sequences):
                fig = Figure(figsize=(fig_width, fig_height))
                canvas = FigureCanvasAgg(fig)

                # Create subplots for acceleration and steering
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                # Plot acceleration sequence
                time_steps = np.arange(len(action_seq)) * self._DT
                accelerations = action_seq[:, 0]
                ax1.plot(
                    time_steps, accelerations, "b-", linewidth=2, label="Acceleration"
                )
                ax1.set_ylabel("Acceleration (m/s²)")
                ax1.set_title(f"Action Sequence {i+1}/{len(self._action_sequences)}")
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Plot steering sequence
                steering_angles = action_seq[:, 1]
                ax2.plot(
                    time_steps, steering_angles, "r-", linewidth=2, label="Steering"
                )
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Steering Angle (rad)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Add timestep indicator
                if i < len(self._action_sequences):
                    current_time = i * self._replan_interval
                    for ax in [ax1, ax2]:
                        ax.axvline(
                            x=current_time,
                            color="green",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label="Current Time",
                        )

                fig.tight_layout()

                # Convert matplotlib figure to numpy array
                canvas.draw()
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
                imgs.append(buf)

                # Clean up
                fig.clear()
                plt.close(fig)

            # Save video
            if imgs:
                mediapy.write_video(action_seq_video_path, imgs, fps=10)
                print(f"Action sequence video saved to '{action_seq_video_path}'")
            else:
                print("No action sequences to save")

        except Exception as e:
            print(f"Failed to save action sequence video: {e}")
