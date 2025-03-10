# Description: Wrapper for Waymax environment to be compatible with skrl
import dataclasses
from typing import Any, Iterator, List, Tuple, Union, override

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
import cv2


def construct_SDC_route(
    state: _env.PlanningAgentSimulatorState,
) -> _env.PlanningAgentSimulatorState:
    """Construct a SDC route from the logged trajectory. This is neccessary for the progression metric as WOMD doesn't release their routes.
    Args:
        state: The simulator state.
    Returns:
        The updated simulator state with the SDC route.
    """
    # Calculate arc lengths (cumulative distances along the trajectory)
    # Select sdc trajectory
    sdc_trajectory: datatypes.Trajectory = datatypes.select_by_onehot(
        state.log_trajectory,
        state.object_metadata.is_sdc,
        keepdims=True,
    )
    x = sdc_trajectory.x
    y = sdc_trajectory.y
    z = sdc_trajectory.z

    # Downsample trajectory coordinates by keeping every 3rd point
    stride = 3

    # Get downsampled coordinates
    x_downsampled = x[..., ::stride]
    y_downsampled = y[..., ::stride]
    z_downsampled = z[..., ::stride]

    # Check if last point needs to be added
    num_points = x.shape[-1]
    last_included = (num_points - 1) % stride == 0

    x = jnp.concatenate([x_downsampled, x[..., -1:]], axis=-1)
    y = jnp.concatenate([y_downsampled, y[..., -1:]], axis=-1)
    z = jnp.concatenate([z_downsampled, z[..., -1:]], axis=-1)

    # Calculate differences between consecutive points
    dx = jnp.diff(x, axis=-1)
    dy = jnp.diff(y, axis=-1)

    # Calculate Euclidean distance for each step
    step_distances = jnp.sqrt(dx**2 + dy**2)

    # Calculate cumulative distances
    arc_lengths = jnp.zeros_like(x)
    arc_lengths = arc_lengths.at[..., 1:].set(jnp.cumsum(step_distances, axis=-1))

    logged_route = datatypes.Paths(
        x=x,
        y=y,
        z=z,
        valid=jnp.array([[True] * len(x[0])]),
        arc_length=arc_lengths,
        on_route=jnp.array([[True]]),
        ids=jnp.array([[0] * len(x)]),  # Dummy ID
    )
    return dataclasses.replace(
        state,
        sdc_paths=logged_route,
    )


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
    state = construct_SDC_route(state)
    observation = env.observe(state).reshape(1, -1)
    return state, observation


merged_reset = jit(merged_reset, static_argnums=(0,))


class WaymaxWrapper(skrl_wrappers.Wrapper):
    def __init__(
        self,
        env: _env.PlanningAgentEnvironment,
        scenario_loader: Iterator[datatypes.SimulatorState],
    ):
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._state: _env.PlanningAgentSimulatorState | None = None
        self._reward: float | None = None

        self._states: List[_env.PlanningAgentSimulatorState] = []  # For rendering
        self._rewards: List[float] = []  # For rendering

    @override
    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        scenario = next(self._scenario_loader)
        self._state, observation = merged_reset(self._env, scenario)
        observation = np.array(observation).reshape(1, -1)
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
        self._state, observation, reward, terminated, truncated = merged_step(
            self._env, self._state, actions  # type: ignore
        )
        observation = np.array(observation).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        terminated = np.array(terminated).reshape(1, -1)
        truncated = np.array(truncated).reshape(1, -1)

        self._reward = reward[0, 0]
        return observation, reward, terminated, truncated, {}

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: np.ndarray or jax.Array
        """
        raise NotImplementedError

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
        action_spec: specs.BoundedArray = self._env.action_spec().data  # type: ignore
        return gymnasium.spaces.Box(
            low=action_spec.minimum, high=action_spec.maximum, shape=(2,), dtype=action_spec.dtype  # type: ignore
        )
