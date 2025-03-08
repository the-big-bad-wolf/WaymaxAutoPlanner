# Description: This file overrides the default Waymax observation function
from typing import Any, Tuple, override

import jax
import jax.numpy as jnp
import waymax.utils.geometry as utils
from dm_env import specs
from waymax import datatypes
from waymax import env as _env
from waymax.env import typedefs as types


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
    ray_dir_x = jnp.sin(ray_angle)
    ray_dir_y = jnp.cos(ray_angle)

    # Calculate segment direction
    segment_dir_x = segment_dirs[:, 0]
    segment_dir_y = segment_dirs[:, 1]

    # Calculate determinant for intersection test
    det = segment_dir_x * ray_dir_y - segment_dir_y * ray_dir_x

    # Avoid division by zero for parallel lines
    is_parallel = jnp.abs(det) < 1e-8
    det = jnp.where(is_parallel, 1.0, det)  # Avoid division by zero

    # Calculate t1 and t2 parameters
    t1 = (start_points[:, 0] * ray_dir_y - start_points[:, 1] * ray_dir_x) / det
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


def find_closest_distance(
    i, initval: Tuple[jax.Array, datatypes.RoadgraphPoints, jax.Array]
):
    circogram, rg_points, ray_angles = initval
    ray_angle = ray_angles[i]

    # Only consider valid points
    candidate_points = rg_points.valid

    # Only consider road edge points
    candidate_points = candidate_points & (
        (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_MEDIAN)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_UNKNOWN)
    )

    # If no valid points, return the default distance
    has_valid = jnp.any(candidate_points)

    # Create line segments from roadgraph points
    starting_points = jnp.stack([rg_points.x, rg_points.y], axis=1)
    dir_xy = jnp.stack([rg_points.dir_x, rg_points.dir_y], axis=1)

    # Calculate intersection distances
    intersection_distances = ray_segment_intersection(
        ray_angle, starting_points, dir_xy
    )
    masked_distances = jnp.where(candidate_points, intersection_distances, 100.0)

    # Find minimum distance among valid points
    min_distance = jnp.min(masked_distances)

    # Update only the i-th element and return the whole array
    circogram = circogram.at[i].set(jnp.where(has_valid, min_distance, circogram[i]))
    return circogram, rg_points, ray_angles


def create_circogram(observation: datatypes.Observation, num_rays: int) -> jax.Array:
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    rg_points = observation.roadgraph_static_points
    # For each ray angle, find the closest intersection
    circogram = jnp.full(num_rays, 100.0)  # Default max distance
    (
        circogram,
        _,
        _,
    ) = jax.lax.fori_loop(
        0,
        num_rays,
        find_closest_distance,
        (circogram, rg_points, ray_angles),
    )
    return circogram


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

        num_rays = 64
        circogram = create_circogram(observation, num_rays)

        obs = jnp.concatenate(
            [
                sdc_xy_goal.flatten(),
                sdc_velocity_xy.flatten(),
                circogram.flatten(),
            ],
            axis=-1,
        )

        # jax.debug.breakpoint()
        return obs

    @override
    def observation_spec(self) -> types.Observation:
        """Returns the observation spec of the environment.
        Returns:
            Observation spec of the environment.
        """
        # Define dimensions for each observation component
        sdc_goal_dim = 2
        sdc_vel_dim = 2
        circogram_dim = 64

        # Total shape is the sum of all component dimensions
        total_dim = sdc_goal_dim + sdc_vel_dim + circogram_dim

        # Define min/max bounds for each component
        sdc_goal_min = [-1000] * sdc_goal_dim
        sdc_goal_max = [1000] * sdc_goal_dim

        sdc_vel_min = [-30] * sdc_vel_dim
        sdc_vel_max = [30] * sdc_vel_dim

        circogram_min = [0] * circogram_dim
        circogram_max = [100] * circogram_dim

        # Combine all bounds
        min_bounds = jnp.array(sdc_goal_min + sdc_vel_min + circogram_min)
        max_bounds = jnp.array(sdc_goal_max + sdc_vel_max + circogram_max)

        return specs.BoundedArray(
            shape=(total_dim,),
            minimum=min_bounds,
            maximum=max_bounds,
            dtype=jnp.float32,
        )
