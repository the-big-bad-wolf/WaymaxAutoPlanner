import dataclasses
from typing import Any, Iterator, List, Tuple, Union, override

import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import skrl.envs.wrappers.jax as skrl_wrappers
import waymax.utils.geometry as utils
from dm_env import specs
from jax import jit
from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.jax import SequentialTrainer
from waymax import agents
from waymax import config as _config
from waymax import dataloader, datatypes, dynamics
from waymax import env as _env
from waymax import visualization
from waymax.env import typedefs as types

# Set the backend to "jax" or "numpy"
config.jax.backend = "numpy"
config.jax.device = jax.devices("cpu")[0]


class WaymaxEnv(_env.PlanningAgentEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def observe(self, state: _env.PlanningAgentSimulatorState) -> Any:
        """Computes the observation for the given simulation state.

        Here we assume that the default observation is just the simulator state. We
        leave this for the user to override in order to provide a user-specific
        observation function. A user can use this to move some of their model
        specific post-processing into the environment rollout in the actor nodes. If
        they want this post-processing on the accelerator, they can keep this the
        same and implement it on the learner side. We provide some helper functions
        at datatypes.observation.py to help write your own observation functions.

        Args:
          state: Current state of the simulator of shape (...).

        Returns:
          Simulator state as an observation without modifications of shape (...).
        """
        # Get base observation first
        observation = datatypes.sdc_observation_from_state(state, roadgraph_top_k=1000)

        sdc_trajectory = datatypes.select_by_onehot(
            observation.trajectory,
            observation.is_ego,
        )
        sdc_velocity_xy = sdc_trajectory.vel_xy
        sdc_xy_goal = datatypes.select_by_onehot(
            state.log_trajectory.xy[..., -1, :],
            state.object_metadata.is_sdc,
            keepdims=True,
        )
        sdc_xy_goal = utils.transform_points(observation.pose2d.matrix, sdc_xy_goal)[0]
        sdc_yaw_goal = datatypes.select_by_onehot(
            state.log_trajectory.yaw[..., -1],
            state.object_metadata.is_sdc,
            keepdims=True,
        )

        _, sdc_idx = jax.lax.top_k(observation.is_ego, k=1)
        non_sdc_xy = jnp.delete(
            observation.trajectory.xy, sdc_idx, axis=1, assume_unique_indices=True
        ).reshape(127, 2)
        non_sdc_vel_xy = jnp.delete(
            observation.trajectory.vel_xy, sdc_idx, axis=1, assume_unique_indices=True
        ).reshape(127, 2)
        non_sdc_valid = jnp.delete(
            observation.trajectory.valid, sdc_idx, axis=1, assume_unique_indices=True
        ).reshape(127, 1)

        # Set positions of invalid objects to 10000
        non_sdc_xy = non_sdc_xy * non_sdc_valid + (1 - non_sdc_valid) * 10000

        # Set velocities of invalid objects to 0
        non_sdc_vel_xy = non_sdc_vel_xy * non_sdc_valid

        rg_points = observation.roadgraph_static_points
        rg_angles = jnp.arctan2(rg_points.x, rg_points.y)
        num_rays = 60
        ray_angles = jnp.linspace(-jnp.pi, jnp.pi, num_rays)

        # For each ray angle, find the closest roadgraph point
        closest_distances = jnp.full(num_rays, 100.0)  # Default large distance

        def find_closest_distance(i, distances):
            ray_angle = ray_angles[i]

            # Calculate angle difference and normalize to [-pi, pi]
            angle_diff = rg_angles - ray_angle
            angle_diff = (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

            # Only consider points roughly in the ray direction (within tolerance)
            angle_tolerance = jnp.pi / num_rays
            candidate_points = jnp.abs(angle_diff) < angle_tolerance

            # Only consider valid points
            candidate_points = candidate_points & rg_points.valid

            # Only consider boundary points
            candidate_points = candidate_points & (
                (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
                | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_MEDIAN)
                | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_UNKNOWN)
            )

            # If no valid points, return the default distance
            has_valid = jnp.any(candidate_points)

            # Calculate distances to all valid points
            point_distances = jnp.sqrt(rg_points.x**2 + rg_points.y**2)
            masked_distances = jnp.where(candidate_points, point_distances, 100.0)

            # Find minimum distance among valid points
            min_distance = jnp.min(masked_distances)

            # Update only the i-th element and return the whole array
            new_distances = distances.at[i].set(
                jnp.where(has_valid, min_distance, distances[i])
            )
            return new_distances

        closest_distances = jax.lax.fori_loop(
            0, num_rays, find_closest_distance, closest_distances
        )

        obs = jnp.concatenate(
            [
                sdc_xy_goal.flatten(),
                sdc_velocity_xy.flatten(),
            ],
            axis=-1,
        )

        # jax.debug.breakpoint()
        return obs

    @override
    def observation_spec(self) -> types.Observation:
        return specs.BoundedArray(
            shape=(4,),
            minimum=jnp.array([-100] * 2 + [-30] * 2),  # Ingen y fart????
            maximum=jnp.array([100] * 2 + [30] * 2),
            dtype=jnp.float32,
        )


def setup_waymax():
    # path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
    path = "./data/training_tfexample.tfrecord@5"
    max_num_objects = 128
    data_loader_config = dataclasses.replace(
        _config.WOD_1_1_0_TRAINING,
        path=path,
        max_num_objects=max_num_objects,
        max_num_rg_points=30000,
    )
    data_iter = dataloader.simulator_state_generator(config=data_loader_config)
    sim_agent_config = _config.SimAgentConfig(
        agent_type=_config.SimAgentType.IDM,
        controlled_objects=_config.ObjectType.NON_SDC,
    )
    metrics_config = _config.MetricsConfig(metrics_to_run=("log_divergence",))
    reward_config = _config.LinearCombinationRewardConfig(
        rewards={"log_divergence": -1.0}
    )
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        metrics=metrics_config,
        rewards=reward_config,
        max_num_objects=max_num_objects,
        sim_agents=[sim_agent_config],
    )
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
    env = WaymaxEnv(
        dynamics_model=dynamics_model,
        config=env_config,
        sim_agent_actors=[agents.create_sim_agents_from_config(sim_agent_config)],
        sim_agent_params=[{}],
    )
    return env, data_iter


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
    ):
        super().__init__(env)
        self._env: _env.PlanningAgentEnvironment
        self._scenario_loader = scenario_loader
        self._states: List[_env.PlanningAgentSimulatorState] = []  # For rendering
        self._state: _env.PlanningAgentSimulatorState | None = None
        print("action_spec", self.action_space)
        print("observation_spec", self.observation_space)

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
        """Render the environment

        :return: Any value from the wrapped environment
        :rtype: any
        """
        self._states.append(self._state)  # type: ignore # store state for video generation

    @override
    def close(self) -> None:
        """Close the environment"""
        if len(self._states) == 0:
            return

        imgs = []
        jit_observe = jit(datatypes.sdc_observation_from_state)
        for state in self._states:
            # observation = jit_observe(state)
            # imgs.append(visualization.plot_observation(observation, 0))
            imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
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


class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=1,
        reduction="sum",
        **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(10)(inputs["states"]))
        x = nn.relu(nn.Dense(10)(x))
        x = nn.Dense(self.num_actions)(x)  # type: ignore
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return nn.tanh(x), log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device=None, clip_actions=False, **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(10)(inputs["states"]))
        x = nn.relu(nn.Dense(10)(x))
        x = nn.Dense(1)(x)
        return x, {}


env, data_iter = setup_waymax()
env = WaymaxWrapper(env, data_iter)

# instantiate a memory as rollout buffer (any memory can be used for this)
mem_size = 4096
memory = RandomMemory(memory_size=mem_size, num_envs=1)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space)
models["value"] = Value(env.observation_space, env.action_space)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = mem_size  # memory_size


agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()


# load the latest checkpoint (adjust the path as needed)
# agent.load("./runs/25-03-04_13-14-37-847444_PPO/checkpoints/best_agent.pickle")
# visualize the training
trainer.timesteps = 1000
trainer.headless = False
trainer.eval()
