# Description: Wrapper for Waymax environment to be compatible with skrl
import glob
import os
import time
from typing import Any, Iterator, List, Tuple, Union, override

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
from waymax import datatypes
from waymax import env as _env
from waymax import visualization

from mpc import get_MPC_action
from sampler import get_best_action, jitted_get_best_action


def merged_step(
    env: _env.PlanningAgentEnvironment,
    state: _env.PlanningAgentSimulatorState,
    actions: Union[np.ndarray, jax.Array],
):
    action = datatypes.Action(data=actions.flatten(), valid=jnp.ones((1,), dtype=bool))  # type: ignore
    new_state = env.step(state, action)
    reward = env.reward(state, action).reshape(1, -1)
    observation = env.observe(new_state).reshape(1, -1)
    terminated = env.termination(new_state).reshape(1, -1)
    truncated = env.truncation(new_state).reshape(1, -1)
    return new_state, observation, reward, terminated, truncated


merged_step = jit(merged_step, static_argnums=(0,))


def merged_reset(
    env: _env.PlanningAgentEnvironment, scenario: datatypes.SimulatorState
):
    state = env.reset(scenario)
    observation = env.observe(state).reshape(1, -1)
    return state, observation


merged_reset = jit(merged_reset, static_argnums=(0,))


def jerk_reward(
    actions: jax.Array | np.ndarray, prev_actions: jax.Array | np.ndarray
) -> jax.Array | np.ndarray:
    """Calculate jerk reward based on the difference between the current and previous actions."""
    accel_jerk = np.abs(actions[0][0] - prev_actions[0]) / 2
    steering_jerk = np.abs(actions[0][1] - prev_actions[1]) / 2
    return -1 * accel_jerk - 5 * steering_jerk


class WaymaxWrapper(skrl_wrappers.Wrapper):
    def __init__(
        self,
        env: _env.PlanningAgentEnvironment,
        scenario_loader: Iterator[datatypes.SimulatorState],
        action_space_type: str = "bicycle",
    ):
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._random_key = jax.random.key(int(time.time()))
        self._action_space_type = action_space_type
        if action_space_type == "trajectory_sampling":
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

        self._MPC_action: Tuple[float, float] = (0.0, 0.0)
        self._prev_action: jax.Array | np.ndarray = np.zeros(
            (2,), dtype=np.float32
        )  # For jerk reward

        self._current_state: _env.PlanningAgentSimulatorState | None = None
        self._current_reward: float | None = None  # For rendering

    @override
    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        scenario = next(self._scenario_loader)
        self._current_state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)

        if self._action_space_type == "trajectory_sampling":
            self._current_action_sequence = jnp.zeros(
                (int(round(self._horizon / self._DT)), 2), dtype=jnp.float32
            )

        """"
        mpc_action = get_MPC_action(self._state)
        self._MPC_action = mpc_action
        mpc_action_array = np.array(mpc_action).reshape(1, 2)  # Convert to 1x2 array
        # Prepend MPC actions to observation
        observation = np.concatenate([mpc_action_array, observation], axis=1)
        """

        return observation, {}

    @override
    def step(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        # Check for NaN values in actions
        if jnp.isnan(actions).any():
            raise ValueError("Actions contain NaN values. Halting execution.")

        actions = actions.flatten()

        if self._action_space_type == "trajectory_sampling":
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

            # Get the best action from the Gaussian polynomial distribution
            self._random_key, subkey = jax.random.split(self._random_key)
            action_sequence = jitted_get_best_action(
                means,
                cholesky_diag,
                cholesky_offdiag,
                self._current_state,
                self._current_action_sequence,
                self._env,
                self._nr_rollouts,
                self._horizon,
                subkey,
            )
            rl_accel = action_sequence[0][0]
            rl_steering = action_sequence[0][1]

            # Shift the action sequence for the next step (receding horizon)
            # Take the sequence from the second element onwards
            shifted_sequence = action_sequence[1:]
            # Create a zero action with the same shape and dtype as one action step
            zero_action = jnp.zeros(
                (1, action_sequence.shape[1]), dtype=action_sequence.dtype
            )
            # Update the stored sequence by appending the zero action
            self._current_action_sequence = jnp.concatenate(
                [shifted_sequence, zero_action], axis=0
            )

        elif self._action_space_type == "bicycle":
            rl_accel, rl_steering = actions

        else:
            raise ValueError(f"Unknown action_space_type: {self._action_space_type}")

        mpc_accel, mpc_steering = self._MPC_action

        combined_accel = rl_accel  # + mpc_accel
        combined_steering = rl_steering  # + mpc_steering

        # Reshape for the environment step
        combined_actions = jnp.array(
            [combined_accel, combined_steering], dtype=np.float32
        )
        self._current_state, observation, reward, terminated, truncated = merged_step(
            self._env, self._current_state, combined_actions  # type: ignore
        )

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)

        # reward += jerk_reward(combined_actions, self._prev_action)
        # self._prev_action = combined_actions[0]  # Update previous action
        self._current_reward = reward[0, 0]
        """
        mpc_action = get_MPC_action(self._state)
        self._MPC_action = mpc_action
        mpc_action_array = np.array(mpc_action).reshape(1, 2)  # Convert to 1x2 array
        # Prepend MPC actions to observation
        observation = np.concatenate([mpc_action_array, observation], axis=1)
        observation = observation.reshape(1, -1)
        """
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
            self._action_sequences.append(self._current_action_sequence.copy())

    @override
    def close(self) -> None:
        """Close the environment"""
        if not self._states:
            return

        # Find the newest folder in runs/

        runs_dir = "runs/"
        os.makedirs(runs_dir, exist_ok=True)
        run_folders = glob.glob(os.path.join(runs_dir, "*"))
        if not run_folders:
            newest_folder = os.path.join(runs_dir, "default")
            os.makedirs(newest_folder, exist_ok=True)
        else:
            newest_folder = max(run_folders, key=os.path.getctime)

        # Create video paths
        waymax_video_path = os.path.join(newest_folder, "waymax.mp4")
        action_seq_video_path = os.path.join(newest_folder, "action_sequences.mp4")

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
                (self._cholesky_diag_dim,), 1e-5, dtype=np.float32
            )
            cholesky_diag_max = np.full(
                (self._cholesky_diag_dim,), 0.4, dtype=np.float32
            )

            cholesky_offdiag_min = np.full(
                (self._cholesky_offdiag_dim,), -0.2, dtype=np.float32
            )
            cholesky_offdiag_max = np.full(
                (self._cholesky_offdiag_dim,), 0.2, dtype=np.float32
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
        elif self._action_space_type == "bicycle":
            # Original simple action space (e.g., acceleration, steering)
            action_spec: specs.BoundedArray = self._env.action_spec().data  # type: ignore
            return gymnasium.spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=(2,),
                dtype=action_spec.dtype,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown action_space_type: {self._action_space_type}")
