# Description: Wrapper for Waymax environment to be compatible with skrl
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
from waymax import datatypes
from waymax import env as _env
from waymax import visualization

from mpc import get_MPC_action
from sampler import jitted_get_best_action


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


class WaymaxWrapper(skrl_wrappers.Wrapper):
    def __init__(
        self,
        env: _env.PlanningAgentEnvironment,
        scenario_loader: Iterator[datatypes.SimulatorState],
        action_space_type: str = "polynomial_trajectory_sampling",
    ):
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._action_space_type = action_space_type
        if action_space_type == "polynomial_trajectory_sampling":
            # Configuration for the polynomial coefficients distribution
            num_polys = 2  # Two cubic polynomials
            poly_degree = 3
            num_coeffs_per_poly = (
                poly_degree + 1
            )  # 4 coefficients per polynomial (a, b, c, d)
            self._mean_dim = (
                num_polys * num_coeffs_per_poly
            )  # 8 dimensions for the mean vector

            # Cholesky factor dimensions for the 8x8 covariance matrix
            cholesky_diag_dim = (
                self._mean_dim  # 8 diagonal elements (parameterized as log-std)
            )
            cholesky_offdiag_dim = (
                self._mean_dim * (self._mean_dim - 1) // 2
            )  # 8*7/2 = 28 off-diagonal elements
            self._cholesky_params_dim = (
                cholesky_diag_dim + cholesky_offdiag_dim
            )  # 8 + 28 = 36 parameters for Cholesky

        self._MPC_action: Tuple[float, float] = (
            0.0,
            0.0,
        )  # For making MPC available as part of observation
        self._prev_action: jax.Array | np.ndarray = np.zeros(
            (2,), dtype=np.float32
        )  # For jerk reward

        self._state: _env.PlanningAgentSimulatorState | None = None  # For rendering
        self._reward: float | None = None  # For rendering
        self._states: List[_env.PlanningAgentSimulatorState] = []  # For rendering
        self._rewards: List[float] = []  # For rendering

    def jerk_reward(
        self, actions: jax.Array | np.ndarray, prev_actions: jax.Array | np.ndarray
    ) -> jax.Array | np.ndarray:
        """Calculate jerk reward based on the difference between the current and previous actions."""
        accel_jerk = np.abs(actions[0][0] - prev_actions[0]) / 2
        steering_jerk = np.abs(actions[0][1] - prev_actions[1]) / 2
        return -1 * accel_jerk - 5 * steering_jerk

    @override
    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        scenario = next(self._scenario_loader)
        self._state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)

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
        actions = actions.flatten()

        if self._action_space_type == "polynomial_trajectory_sampling":
            # Extract the mean and Cholesky parameters from the action vector
            mean_params = jnp.array(actions[: self._mean_dim])
            cholesky_params = jnp.array(actions[self._mean_dim :])

            # Get the best action from the Gaussian polynomial distribution
            rl_accel, rl_steering = jitted_get_best_action(
                mean_params, cholesky_params, self._state, self._env, 50
            )

        elif self._action_space_type == "bicycle":
            rl_accel, rl_steering = actions

        else:
            raise ValueError(f"Unknown action_space_type: {self._action_space_type}")

        mpc_accel, mpc_steering = self._MPC_action

        combined_accel = rl_accel  # + mpc_accel
        combined_steering = rl_steering  # + mpc_steering

        # Reshape for the environment step
        combined_actions = np.array(
            [[combined_accel, combined_steering]], dtype=np.float32
        )
        self._state, observation, reward, terminated, truncated = merged_step(
            self._env, self._state, combined_actions  # type: ignore
        )

        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)

        # reward += self.jerk_reward(combined_actions, self._prev_action)
        # self._prev_action = combined_actions[0]  # Update previous action
        self._reward = reward[0, 0]
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
        self._states.append(self._state)  # type: ignore
        self._rewards.append(self._reward)  # type: ignore

    @override
    def close(self) -> None:
        """Close the environment"""
        if len(self._states) == 0:
            return

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
        mediapy.write_video("./waymax.mp4", imgs, fps=10)
        self._states.clear()
        self._rewards.clear()

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

        if self._action_space_type == "polynomial_trajectory_sampling":
            # Total dimension of the action space vector
            total_dim = self._mean_dim + self._cholesky_params_dim  # 8 + 36 = 44

            # Define bounds for the action space components.
            bound_limit = 10.0
            min_bounds = np.full((total_dim,), -bound_limit, dtype=np.float32)
            max_bounds = np.full((total_dim,), bound_limit, dtype=np.float32)

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
