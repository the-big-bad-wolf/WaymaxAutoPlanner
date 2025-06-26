# Description: This file overrides the default Waymax observation function
import dataclasses
from typing import Any, override

import jax
import jax.numpy as jnp
import waymax.utils.geometry as utils
from dm_env import specs
from waymax import datatypes
from waymax import env as _env
from waymax.env import typedefs as types
from waymax.metrics.roadgraph import is_offroad
from circogram import create_road_circogram, create_object_circogram
from circogram import MAX_CIRCOGRAM_DIST

NUM_RAYS = 64  # Number of rays for the circogram


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

    # Downsample trajectory coordinates
    stride = 5  # Downsample every 5th point

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


class WaymaxEnv(_env.PlanningAgentEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def reset(
        self, state: datatypes.SimulatorState, rng: jax.Array | None = None
    ) -> _env.PlanningAgentSimulatorState:
        """Resets the environment to the given state."""
        # Construct SDC route from logged trajectory
        state = super().reset(state, rng)
        state = construct_SDC_route(state)
        return state

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
        # Get base observation from SDC perspective first
        observation = datatypes.sdc_observation_from_state(state, roadgraph_top_k=1000)
        sdc_trajectory = datatypes.select_by_onehot(
            observation.trajectory,
            observation.is_ego,
            keepdims=True,
        )
        sdc_velocity = sdc_trajectory.vel_x[..., 0, 0]
        sdc_length = sdc_trajectory.length[..., 0, 0]
        sdc_width = sdc_trajectory.width[..., 0, 0]

        # Create the goal position from the last point in the logged trajectory
        sdc_xy_goal = datatypes.select_by_onehot(
            state.log_trajectory.xy[..., -1, :],
            state.object_metadata.is_sdc,
            keepdims=True,
        )
        sdc_xy_goal = utils.transform_points(observation.pose2d.matrix, sdc_xy_goal)[0]
        # Convert the goal position from Cartesian to polar coordinates
        sdc_goal_distance = jnp.sqrt(
            sdc_xy_goal[..., 0] ** 2 + sdc_xy_goal[..., 1] ** 2
        )
        sdc_goal_angle = jnp.arctan2(sdc_xy_goal[..., 1], sdc_xy_goal[..., 0])

        sdc_offroad = is_offroad(sdc_trajectory, observation.roadgraph_static_points)
        sdc_offroad = sdc_offroad.astype(jnp.float32)  # Convert boolean to float32

        road_circogram, road_radial, road_tangential = create_road_circogram(
            observation, NUM_RAYS
        )
        object_circogram, object_radial, object_tangential = create_object_circogram(
            observation, NUM_RAYS
        )

        # Take the minimum of road and object circograms at each angle
        total_circogram = jnp.minimum(road_circogram, object_circogram)
        # Determine which circogram (road or object) provided the minimum distance for each ray
        object_is_closer = object_circogram < road_circogram

        # Select the radial speed corresponding to the minimum distance
        # If object is closer, use object_radial, otherwise use road_radial (which is 0)
        total_radial_speed = jnp.where(object_is_closer, object_radial, road_radial)

        # Select the tangential speed corresponding to the minimum distance
        # If object is closer, use object_tangential, otherwise use road_tangential (which is 0)
        total_tangential_speed = jnp.where(
            object_is_closer, object_tangential, road_tangential
        )

        obs = jnp.concatenate(
            [
                total_circogram.flatten(),
                total_radial_speed.flatten(),
                total_tangential_speed.flatten(),
                sdc_goal_angle.flatten(),
                sdc_goal_distance.flatten(),
                sdc_velocity.flatten(),
                sdc_offroad.flatten(),
                sdc_length.flatten(),
                sdc_width.flatten(),
            ],
            axis=-1,
        )
        return obs

    @override
    def observation_spec(self) -> types.Observation:
        """Returns the observation spec of the environment.
        Returns:
            Observation spec of the environment.
        """
        # Define dimensions for each observation component based on observe method concatenation
        circogram_dim = NUM_RAYS
        radial_speed_dim = NUM_RAYS
        tangential_speed_dim = NUM_RAYS
        sdc_goal_angle_dim = 1
        sdc_goal_distance_dim = 1
        sdc_velocity_dim = 1
        sdc_offroad_dim = 1
        sdc_length_dim = 1
        sdc_width_dim = 1

        # Total shape is the sum of all component dimensions
        total_dim = (
            circogram_dim
            + radial_speed_dim
            + tangential_speed_dim
            + sdc_goal_angle_dim
            + sdc_goal_distance_dim
            + sdc_velocity_dim
            + sdc_offroad_dim
            + sdc_length_dim
            + sdc_width_dim
        )

        # Define bounds for each component
        circogram_min = [0.0] * circogram_dim
        circogram_max = [MAX_CIRCOGRAM_DIST] * circogram_dim

        # Speeds can be positive or negative, assuming max absolute speed of 30 m/s
        radial_speed_min = [-30.0] * radial_speed_dim
        radial_speed_max = [30.0] * radial_speed_dim
        tangential_speed_min = [-30.0] * tangential_speed_dim
        tangential_speed_max = [30.0] * tangential_speed_dim

        sdc_goal_angle_min = [-jnp.pi]
        sdc_goal_angle_max = [jnp.pi]
        sdc_goal_distance_min = [0.0]
        sdc_goal_distance_max = [250.0]  # Max distance observed

        # SDC velocity bounds
        sdc_velocity_min = [-30.0]
        sdc_velocity_max = [30.0]  # Assuming max absolute speed of 30 m/s

        sdc_offroad_min = [0.0]
        sdc_offroad_max = [1.0]

        # Add bounds for vehicle dimensions
        sdc_length_min = [1.0]
        sdc_length_max = [5.2860003]
        sdc_width_min = [1.0]
        sdc_width_max = [2.332]

        # Combine all bounds in the order of concatenation in the observe method
        min_bounds = jnp.array(
            circogram_min
            + radial_speed_min
            + tangential_speed_min
            + sdc_goal_angle_min
            + sdc_goal_distance_min
            + sdc_velocity_min
            + sdc_offroad_min
            + sdc_length_min
            + sdc_width_min,
            dtype=jnp.float32,
        )
        max_bounds = jnp.array(
            circogram_max
            + radial_speed_max
            + tangential_speed_max
            + sdc_goal_angle_max
            + sdc_goal_distance_max
            + sdc_velocity_max
            + sdc_offroad_max
            + sdc_length_max
            + sdc_width_max,
            dtype=jnp.float32,
        )

        return specs.BoundedArray(
            shape=(total_dim,),
            minimum=min_bounds,
            maximum=max_bounds,
            dtype=jnp.float32,
        )
