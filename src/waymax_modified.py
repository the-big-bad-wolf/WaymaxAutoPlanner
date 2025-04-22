# Description: This file overrides the default Waymax observation function
import dataclasses
from typing import Any, Tuple, override

import jax
import jax.numpy as jnp
import waymax.utils.geometry as utils
from dm_env import specs
from waymax import datatypes
from waymax import env as _env
from waymax.env import typedefs as types
from waymax.metrics.roadgraph import is_offroad


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


def ray_segment_intersection(
    ray_angle: jax.Array, start_points: jax.Array, segment_dirs: jax.Array
) -> jax.Array:
    """
    Calculate the intersection distances between a ray and line segments.

    Args:
        ray_angle: The angle of the ray (in radians).
        start_points: Array of shape (N, 2) for segment start points (x,y).
        segment_dirs: Array of shape (N, 2) for segment directions (dx, dy).

    Returns:
        Array of distances from origin to intersections. Returns 100.0 if no intersection.
    """
    # Calculate ray direction
    ray_dir_x = jnp.cos(ray_angle)
    ray_dir_y = jnp.sin(ray_angle)

    # Calculate segment direction
    segment_dir_x = segment_dirs[:, 0]
    segment_dir_y = segment_dirs[:, 1]

    # Calculate determinant for intersection test
    det = segment_dir_x * ray_dir_y - segment_dir_y * ray_dir_x

    # Avoid division by zero for parallel lines
    is_parallel = jnp.abs(det) < 1e-8
    det = jnp.where(is_parallel, 1.0, det)  # Avoid division by zero

    # Calculate t1 and t2 parameters
    t1 = -(start_points[:, 0] * ray_dir_y - start_points[:, 1] * ray_dir_x) / det
    t2 = (start_points[:, 0] * segment_dir_y - start_points[:, 1] * segment_dir_x) / det

    # Check if intersection is within segment (0 <= t1 <= 1) and ray (t2 >= 0)
    valid_t1 = (t1 >= 0.0) & (t1 <= 1.0)
    valid_t2 = t2 >= 0.0

    # Combine intersection validity checks
    valid_intersection = valid_t1 & valid_t2 & ~is_parallel

    # Calculate intersection points and distances
    ix = start_points[:, 0] + t1 * segment_dir_x
    iy = start_points[:, 1] + t1 * segment_dir_y

    # Distance from origin to intersection point
    distances = jnp.sqrt(ix**2 + iy**2)

    # Make sure distances over 100 are not valid
    valid_intersection = valid_intersection & (distances < 100.0)

    # Return distance if valid intersection, otherwise 100.0
    return jnp.where(valid_intersection, distances, 100.0)


def circogram_subroutine(
    i: int, initval: Tuple[jax.Array, jax.Array, Tuple[jax.Array, jax.Array], jax.Array]
):
    circogram, ray_angles, (starting_points, dir_xy), candidate_mask = initval
    ray_angle = ray_angles[i]

    # Calculate intersection distances
    intersection_distances = ray_segment_intersection(
        ray_angle, starting_points, dir_xy
    )

    # Only consider specified segments
    masked_distances = jnp.where(candidate_mask, intersection_distances, 100.0)

    # Find minimum distance among candidate segments
    min_distance = jnp.min(masked_distances)

    # Update only the i-th circogram ray and return the whole array
    circogram = circogram.at[i].set(min_distance)
    return circogram, ray_angles, (starting_points, dir_xy), candidate_mask


def create_road_circogram(
    observation: datatypes.Observation, num_rays: int
) -> jax.Array:
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    circogram = jnp.full(num_rays, 100.0)  # Default max distance
    rg_points = observation.roadgraph_static_points
    candidate_mask = rg_points.valid
    candidate_mask = candidate_mask & (
        (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_MEDIAN)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_UNKNOWN)
    )
    # Create line segments from roadgraph points
    starting_points = jnp.stack([rg_points.x, rg_points.y], axis=1)
    dir_xy = jnp.stack([rg_points.dir_x, rg_points.dir_y], axis=1)
    line_segments = (starting_points, dir_xy)

    (circogram, _, _, _) = jax.lax.fori_loop(
        0,
        num_rays,
        circogram_subroutine,
        (circogram, ray_angles, line_segments, candidate_mask),
    )
    return circogram


def create_object_circogram(
    observation: datatypes.Observation, num_rays: int
) -> jax.Array:
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    circogram = jnp.full(num_rays, 100.0)  # Default max distance

    candidate_mask = observation.trajectory.valid[..., 0, :, 0]
    candidate_mask = candidate_mask & ~observation.is_ego[..., 0, :]
    candidate_mask = jnp.repeat(
        candidate_mask, 4
    )  # (num_objects*4,) Each object has 4 segments

    # Create line segments from object bounding box corners
    obj_corners = observation.trajectory.bbox_corners[0, :, 0, :, :]
    start_indices = jnp.array([0, 1, 2, 3])
    end_indices = jnp.array([1, 2, 3, 0])
    start_points = obj_corners[:, start_indices]  # (num_objects, 4, 2)
    end_points = obj_corners[:, end_indices]  # (num_objects, 4, 2)
    segment_dirs = end_points - start_points  # (num_objects, 4, 2)
    start_points = start_points.reshape(-1, 2)  # (num_objects*4, 2)
    segment_dirs = segment_dirs.reshape(-1, 2)  # (num_objects*4, 2)
    line_segments = (start_points, segment_dirs)

    (circogram, _, _, _) = jax.lax.fori_loop(
        0,
        num_rays,
        circogram_subroutine,
        (circogram, ray_angles, line_segments, candidate_mask),
    )
    return circogram


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
        sdc_velocity_xy = sdc_trajectory.vel_xy

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
        sdc_yaw_goal = datatypes.select_by_onehot(
            state.log_trajectory.yaw[..., -1],
            state.object_metadata.is_sdc,
            keepdims=True,
        )

        sdc_offroad = is_offroad(sdc_trajectory, observation.roadgraph_static_points)
        sdc_offroad = sdc_offroad.astype(jnp.float32)  # Convert boolean to float32

        num_rays = 64
        road_circogram = create_road_circogram(observation, num_rays)
        object_circogram = create_object_circogram(observation, num_rays)

        obs = jnp.concatenate(
            [
                road_circogram.flatten(),
                object_circogram.flatten(),
                sdc_goal_angle.flatten(),
                sdc_goal_distance.flatten(),
                sdc_velocity_xy.flatten(),
                sdc_offroad.flatten(),
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
        # Define dimensions for each observation component
        road_circogram_dim = 64
        object_circogram_dim = 64
        sdc_goal_angle_dim = 1
        sdc_goal_distance_dim = 1
        sdc_vel_dim = 2
        sdc_offroad_dim = 1
        MPC_action_dim = 2

        # Total shape is the sum of all component dimensions
        total_dim = (
            road_circogram_dim
            + object_circogram_dim
            + sdc_goal_angle_dim
            + sdc_goal_distance_dim
            + sdc_vel_dim
            + sdc_offroad_dim
        )

        # Define min/max bounds for each component
        MPC_action_min = [-1.0, -1.0]
        MPC_action_max = [1.0, 1.0]

        sdc_goal_angle_min = [-jnp.pi]
        sdc_goal_angle_max = [jnp.pi]
        sdc_goal_distance_min = [0]
        sdc_goal_distance_max = [250]

        # Radial speed
        sdc_vel_x_min = [-30]
        sdc_vel_x_max = [30]

        # Tangential speed. The limit is calculated based on a maximum absolute speed of 30 m/s
        sdc_vel_y_min = [-9]
        sdc_vel_y_max = [9]

        sdc_offroad_min = [0]
        sdc_offroad_max = [1]

        road_circogram_min = [0] * road_circogram_dim
        road_circogram_max = [100] * road_circogram_dim
        object_circogram_min = [0] * object_circogram_dim
        object_circogram_max = [100] * object_circogram_dim

        # Combine all bounds
        min_bounds = jnp.array(
            road_circogram_min
            + object_circogram_min
            + sdc_goal_angle_min
            + sdc_goal_distance_min
            + sdc_vel_x_min
            + sdc_vel_y_min
            + sdc_offroad_min
        )
        max_bounds = jnp.array(
            road_circogram_max
            + object_circogram_max
            + sdc_goal_angle_max
            + sdc_goal_distance_max
            + sdc_vel_x_max
            + sdc_vel_y_max
            + sdc_offroad_max
        )

        return specs.BoundedArray(
            shape=(total_dim,),
            minimum=min_bounds,
            maximum=max_bounds,
            dtype=jnp.float32,
        )
