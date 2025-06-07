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
    """Performs a multi-step rollout using Waymax's native rollout functionality."""

    # Define simplified actor logic with less redundant operations
    def init_fn(rng, state):
        return {"action_index": 0}

    def select_action(params, state, actor_state, rng):
        # Get current action directly without redundant slicing/squeezing
        action_index = actor_state["action_index"]
        action = params["action_sequence"][action_index]

        # Create action object directly without redundant array creation
        action_obj = datatypes.Action(data=action, valid=jnp.ones(1, dtype=bool))

        # Increment with a simpler expression
        actor_state = {"action_index": jnp.minimum(action_index + 1, num_steps - 1)}

        return agents.WaymaxActorOutput(
            actor_state=actor_state,
            action=action_obj,
            is_controlled=state.object_metadata.is_sdc,
        )

    # Create actor with simplified parameters
    sequence_actor = agents.actor_core_factory(init_fn, select_action)
    actor_params = {"action_sequence": actions}

    # Run rollout with static key to avoid redundant key generation
    rollout_output = _env.rollout(
        scenario=state,
        actor=sequence_actor,
        env=env,
        rng=jax.random.PRNGKey(0),  # Deterministic key because not needed
        rollout_num_steps=num_steps,
        actor_params=actor_params,
    )

    # Extract final state directly
    final_state = jax.tree_map(lambda x: x[-1], rollout_output.state)
    metrics_history = rollout_output.metrics

    # Simplify overlap/offroad detection with one jnp.any call
    overlap_detected = jnp.any(metrics_history["overlap"].value > 0).astype(jnp.int32)
    offroad_detected = jnp.any(metrics_history["offroad"].value > 0).astype(jnp.int32)

    # Get progression data directly
    progression_values = metrics_history["sdc_progression"].value
    rollout_progression = progression_values[-1]

    # Get reward and reshape in one operation
    total_reward = jnp.sum(rollout_output.reward).reshape(1, -1)

    # Get observation and reshape in one operation
    observation = env.observe(final_state).reshape(1, -1)
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
    ):
        """
        Initialize the Waymax wrapper.

        Args:
            env: The Waymax planning agent environment.
            scenario_loader: Iterator providing simulator states.
            action_space_type: One of "bicycle", "bicycle_mpc", or "trajectory_sampling".
        """
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._random_key = jax.random.key(int(time.time()))
        self._action_space_type = action_space_type
        self._prev_action = jnp.zeros((2,), dtype=np.float32)  # For jerk reward

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
            metrics_to_run=("sdc_progression", "overlap")
        )
        reward_config = _config.LinearCombinationRewardConfig(
            rewards={"sdc_progression": 1.0, "overlap": -2.0},
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
        self._replan_interval = 0.1  # seconds

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
        """Step implementation for trajectory sampling action space"""
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

        multistep = int(round(self._replan_interval / self._DT))

        # Perform a multi-step rollout with the action sequence and track events
        (
            self._current_state,
            observation,
            reward,
            terminated,
            truncated,
            had_overlap,
            had_offroad,
            progression,
        ) = merged_multistep(
            self._env,
            self._current_state,
            action_sequence,
            multistep,
        )

        # Update episode tracking flags using the accumulated events from all steps
        self._current_episode_had_overlap = jnp.maximum(
            self._current_episode_had_overlap, had_overlap
        )
        self._current_episode_had_offroad = jnp.maximum(
            self._current_episode_had_offroad, had_offroad
        )

        # Store the current progression
        self._current_episode_progression = progression

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)
        self._current_reward = reward[0, 0]

        # Shift the action sequence for the next step (receding horizon)
        # Take the sequence from the second element onwards
        shifted_sequence = action_sequence[multistep:]
        # Create a zero action with the same shape and dtype as one action step
        zero_action = jnp.zeros(
            (multistep, action_sequence.shape[1]), dtype=action_sequence.dtype
        )
        # Update the stored sequence by appending the zero action
        self._current_action_plan = jnp.concatenate(
            [shifted_sequence, zero_action], axis=0
        )

        return observation, reward, terminated, truncated, {}

    @override
    def render(self, *args, **kwargs) -> Any:
        """Store the state for video generation on close"""
        self._states = getattr(self, "_states", [])
        self._rewards = getattr(self, "_rewards", [])
        # Store the current state and reward for rendering
        self._states.append(self._current_state)
        self._rewards.append(self._current_reward)
        # Store the current action sequence for visualization if needed
        if self._action_space_type == "trajectory_sampling":
            self._action_sequences = getattr(self, "_action_sequences", [])
            self._action_sequences.append(self._current_action_plan.copy())

    @override
    def close(self) -> None:
        """Close the environment"""
        # Determine the correct directory for saving files
        runs_dir = "runs/"
        os.makedirs(runs_dir, exist_ok=True)

        # Check if we're in evaluation mode by looking for eval_logs in recent directories
        run_folders = glob.glob(os.path.join(runs_dir, "*"))
        eval_folders = glob.glob(os.path.join(runs_dir, "*/eval_logs"))

        if eval_folders:
            # Use the most recent eval_logs folder
            newest_eval_folder = max(eval_folders, key=os.path.getctime)
            save_directory = newest_eval_folder
            print(f"Saving to evaluation directory: {save_directory}")
        else:
            # Fall back to the newest run folder
            if not run_folders:
                newest_folder = os.path.join(runs_dir, "default")
                os.makedirs(newest_folder, exist_ok=True)
            else:
                newest_folder = max(run_folders, key=os.path.getctime)
            save_directory = newest_folder
            print(f"Saving to training directory: {save_directory}")

        # Write episode statistics to a file
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

        # Continue with existing video creation code
        if not hasattr(self, "_states") or not self._states:
            return

        # Create video paths in the same directory
        waymax_video_path = os.path.join(save_directory, "waymax.mp4")
        action_seq_video_path = os.path.join(save_directory, "action_sequences.mp4")

        imgs = []
        jit_observe = jit(datatypes.sdc_observation_from_state)
        for i in range(len(self._states)):
            # observation = jit_observe(state)
            # imgs.append(visualization.plot_observation(observation, 0))
            img = visualization.plot_simulator_state(
                self._states[i], use_log_traj=False
            )
            # Add reward text overlay to the bottom left cornerstate
            # Format the reward text
            reward_text = f"Reward: {self._rewards[i]:.4f}"
            # Create a copy to avoid modifying the original
            img_with_text = img.copy()
            # Get image dimensions
            h, w = img_with_text.shape[:2]
            # Draw a semi-transparent black box at the bottom left
            overlay = img_with_text.copy()

            # Draw a semi-transparent black box at the bottom right
            text_size = cv2.getTextSize(reward_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                0
            ]
            text_width = text_size[0] + 10  # Add padding
            pt1 = (w - text_width - 10, h - 40)  # Top-left corner of the box
            pt2 = (w - 10, h - 10)  # Bottom-right corner of the box
            cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)  # Black filled rectangle
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, img_with_text, 1 - alpha, 0, img_with_text)
            # Add text on top of the box
            cv2.putText(
                img_with_text,
                reward_text,
                (w - text_width - 5, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            img = img_with_text
            imgs.append(img)
        mediapy.write_video(waymax_video_path, imgs, fps=10)
        self._states.clear()
        self._rewards.clear()

        if hasattr(self, "_action_sequences") and len(self._action_sequences) > 0:
            try:
                import matplotlib.pyplot as plt

                action_seq_imgs = []
                horizon = self._action_sequences[0].shape[
                    0
                ]  # Number of timesteps in sequence
                time_steps = np.arange(horizon) * 0.1  # Assuming 10Hz (0.1s timestep)

                # Determine appropriate y-axis limits
                all_values = np.concatenate(
                    [seq.flatten() for seq in self._action_sequences]
                )
                min_val, max_val = np.min(all_values), np.max(all_values)
                y_margin = (max_val - min_val) * 0.1  # 10% margin
                y_min, y_max = min_val - y_margin, max_val + y_margin

                for i in range(len(self._action_sequences)):
                    # Create a figure for this timestep's action sequence
                    fig = Figure(figsize=(10, 6))
                    canvas = FigureCanvasAgg(fig)
                    ax = fig.add_subplot(111)

                    # Plot acceleration and steering sequences
                    ax.plot(
                        time_steps,
                        self._action_sequences[i][:, 0],
                        "b-",
                        label="Acceleration",
                    )
                    ax.plot(
                        time_steps,
                        self._action_sequences[i][:, 1],
                        "r-",
                        label="Steering",
                    )

                    # Add labels and title
                    ax.set_xlabel("Horizon Time (s)")
                    ax.set_ylabel("Action Value")
                    ax.set_title(f"Planned Action Sequence - Step {i}")
                    if i < len(self._rewards):
                        ax.text(
                            0.02,
                            0.95,
                            f"Reward: {self._rewards[i]:.4f}",
                            transform=ax.transAxes,
                            bbox=dict(facecolor="white", alpha=0.5),
                        )
                    ax.legend()
                    ax.grid(True)

                    # Set y-axis limits for consistent visualization
                    ax.set_ylim(y_min, y_max)

                    # Convert figure to image
                    canvas.draw()
                    img = np.array(canvas.renderer.buffer_rgba())
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    action_seq_imgs.append(img)

                    # Close the figure to free memory
                    plt.close(fig)

                # Write video of action sequences
                if action_seq_imgs:
                    mediapy.write_video(action_seq_video_path, action_seq_imgs, fps=10)
                    print(f"Action sequence video saved to '{action_seq_video_path}'")
            except Exception as e:
                print(f"Failed to generate action sequence video: {e}")

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
