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

NUM_RAYS = 64  # Number of rays for the circogram
MAX_CIRCOGRAM_DIST = 60.0  # Maximum distance for the circogram (meters)
Z_TOLERANCE = 3.0  # Filter out objects and road graph points that are on a different height (meters)


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
        Array of distances from origin to intersections. Returns MAX_CIRCOGRAM_DIST if no intersection.
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

    # Make sure distances over MAX_CIRCOGRAM_DIST are not valid
    valid_intersection = valid_intersection & (distances < MAX_CIRCOGRAM_DIST)

    # Return distance if valid intersection, otherwise MAX_CIRCOGRAM_DIST
    return jnp.where(valid_intersection, distances, MAX_CIRCOGRAM_DIST)


def circogram_subroutine(
    i: int,
    initval: Tuple[
        jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array], jax.Array
    ],
):
    (
        circogram,
        winning_indices,
        ray_angles,
        (starting_points, dir_xy),
        candidate_mask,
    ) = initval
    ray_angle = ray_angles[i]

    # Calculate intersection distances
    intersection_distances = ray_segment_intersection(
        ray_angle, starting_points, dir_xy
    )

    # Only consider specified segments
    masked_distances = jnp.where(
        candidate_mask, intersection_distances, MAX_CIRCOGRAM_DIST
    )

    # Find minimum distance and index among candidate segments
    min_distance = jnp.min(masked_distances)
    # Use argmin, handle case where min is MAX_CIRCOGRAM_DIST (no valid hit)
    winning_idx = jnp.argmin(masked_distances)
    # If min_distance is MAX_CIRCOGRAM_DIST, set index to -1
    winning_idx = jnp.where(min_distance >= MAX_CIRCOGRAM_DIST, -1, winning_idx)

    # Update circogram ray and winning index
    circogram = circogram.at[i].set(min_distance)
    winning_indices = winning_indices.at[i].set(winning_idx)

    return (
        circogram,
        winning_indices,
        ray_angles,
        (starting_points, dir_xy),
        candidate_mask,
    )


def create_road_circogram(
    observation: datatypes.Observation, num_rays: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Calculates the distances to the nearest road edge along rays.

    Args:
        observation: The observation data containing roadgraph information.
        num_rays: The number of rays to cast for the circogram.

    Returns:
        A tuple containing:
            - circogram: Array of distances to the nearest road edge for each ray.
            - ray_radial_speed: Array of radial speeds (always 0 for static road edges).
            - ray_tangential_speed: Array of tangential speeds (always 0 for static road edges).
    """
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    circogram = jnp.full(num_rays, MAX_CIRCOGRAM_DIST)  # Default max distance
    winning_segment_indices = jnp.full(
        num_rays, -1, dtype=jnp.int32
    )  # Initialize winning indices

    rg_points = observation.roadgraph_static_points
    candidate_mask = rg_points.valid
    candidate_mask = candidate_mask & (
        (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_MEDIAN)
        | (rg_points.types == datatypes.MapElementIds.ROAD_EDGE_UNKNOWN)
    )
    sdc_trajectory = datatypes.select_by_onehot(
        observation.trajectory,
        observation.is_ego,
        keepdims=False,
    )
    # Get SDC's current z-coordinate
    sdc_z = sdc_trajectory.z[..., 0, 0]

    z_diff = jnp.abs(rg_points.z - sdc_z)
    on_same_plane = z_diff <= Z_TOLERANCE
    candidate_mask = candidate_mask & on_same_plane

    # Create line segments from roadgraph points
    starting_points = jnp.stack([rg_points.x, rg_points.y], axis=1)
    dir_xy = jnp.stack([rg_points.dir_x, rg_points.dir_y], axis=1)
    line_segments = (starting_points, dir_xy)

    # Run loop with subroutine
    (circogram, _, _, _, _) = jax.lax.fori_loop(
        0,
        num_rays,
        circogram_subroutine,
        (circogram, winning_segment_indices, ray_angles, line_segments, candidate_mask),
    )

    # Road edges are static, so their velocities are always zero.
    # We return zero speeds for consistency with create_object_circogram's output signature.
    ray_radial_speed = jnp.zeros(num_rays)
    ray_tangential_speed = jnp.zeros(num_rays)

    return circogram, ray_radial_speed, ray_tangential_speed


def create_object_circogram(
    observation: datatypes.Observation, num_rays: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    circogram = jnp.full(num_rays, MAX_CIRCOGRAM_DIST)  # Default max distance
    winning_segment_indices = jnp.full(num_rays, -1, dtype=jnp.int32)

    # --- Prepare segments and mask ---
    candidate_mask = observation.trajectory.valid[..., 0, :, 0]
    candidate_mask = candidate_mask & ~observation.is_ego[..., 0, :]
    # Filter objects based on z-plane
    sdc_trajectory = datatypes.select_by_onehot(
        observation.trajectory,
        observation.is_ego,
        keepdims=False,
    )
    # Get SDC's current z-coordinate
    sdc_z = sdc_trajectory.z[..., 0, 0]
    # Get z-coordinates of all objects at the current timestep
    obj_z = observation.trajectory.z[..., 0, :, 0]  # Shape: (batch, num_objects)
    z_diff = jnp.abs(obj_z - sdc_z)
    on_same_plane = z_diff <= Z_TOLERANCE
    candidate_mask = candidate_mask & on_same_plane
    candidate_mask = jnp.repeat(
        candidate_mask, 4
    )  # (num_objects*4,) Each object has 4 segments

    obj_corners = observation.trajectory.bbox_corners[0, :, 0, :, :]
    start_indices = jnp.array([0, 1, 2, 3])
    end_indices = jnp.array([1, 2, 3, 0])
    start_points = obj_corners[:, start_indices]
    end_points = obj_corners[:, end_indices]
    segment_dirs = end_points - start_points
    start_points = start_points.reshape(-1, 2)
    segment_dirs = segment_dirs.reshape(-1, 2)
    line_segments = (start_points, segment_dirs)
    # --- End segment prep ---

    # --- Run loop with subroutine ---
    (circogram, winning_segment_indices, _, _, _) = jax.lax.fori_loop(
        0,
        num_rays,
        circogram_subroutine,
        (circogram, winning_segment_indices, ray_angles, line_segments, candidate_mask),
    )
    # --- End loop ---

    # --- Map winning segment indices to velocities ---
    object_indices = winning_segment_indices // 4  # Get object index from segment index
    obj_vel_xy = observation.trajectory.vel_xy[0, :, 0, :]  # Shape: (num_objects, 2)

    # Gather velocities based on object_indices, handle -1 for no hit
    valid_object_hit = winning_segment_indices >= 0
    # Use index 0 for invalid hits temporarily, mask results later
    valid_object_indices = jnp.where(valid_object_hit, object_indices, 0)
    hit_velocities = obj_vel_xy[valid_object_indices]  # Shape: (num_rays, 2)

    # --- End velocity mapping ---

    # --- Calculate Polar Velocities ---
    # Get ray direction vectors
    ray_dir_x = jnp.cos(ray_angles)
    ray_dir_y = jnp.sin(ray_angles)

    # Project hit velocities onto ray direction (radial speed)
    radial_speed = hit_velocities[:, 0] * ray_dir_x + hit_velocities[:, 1] * ray_dir_y

    # Project hit velocities onto direction perpendicular to the ray (tangential speed)
    # Perpendicular vector: (-ray_dir_y, ray_dir_x)
    tangential_speed = (
        -hit_velocities[:, 0] * ray_dir_y + hit_velocities[:, 1] * ray_dir_x
    )

    # Apply mask for valid hits, setting speeds to 0.0 for non-hits
    ray_radial_speed = jnp.where(valid_object_hit, radial_speed, 0.0)
    ray_tangential_speed = jnp.where(valid_object_hit, tangential_speed, 0.0)
    # --- End Polar Velocity Calculation ---

    return circogram, ray_radial_speed, ray_tangential_speed


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
        # Define dimensions for each observation component based on observe method concatenation
        circogram_dim = NUM_RAYS
        radial_speed_dim = NUM_RAYS
        tangential_speed_dim = NUM_RAYS
        sdc_goal_angle_dim = 1
        sdc_goal_distance_dim = 1
        sdc_vel_dim = 2  # (vx, vy)
        sdc_offroad_dim = 1

        # Total shape is the sum of all component dimensions
        total_dim = (
            circogram_dim
            + radial_speed_dim
            + tangential_speed_dim
            + sdc_goal_angle_dim
            + sdc_goal_distance_dim
            + sdc_vel_dim
            + sdc_offroad_dim
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
        sdc_vel_x_min = [-30.0]
        sdc_vel_x_max = [30.0]
        sdc_vel_y_min = [-9.0]  # Estimated max tangential speed is 9 m/s
        sdc_vel_y_max = [9.0]

        sdc_offroad_min = [0.0]
        sdc_offroad_max = [1.0]

        # Combine all bounds in the order of concatenation in the observe method
        min_bounds = jnp.array(
            circogram_min
            + radial_speed_min
            + tangential_speed_min
            + sdc_goal_angle_min
            + sdc_goal_distance_min
            + sdc_vel_x_min
            + sdc_vel_y_min
            + sdc_offroad_min,
            dtype=jnp.float32,
        )
        max_bounds = jnp.array(
            circogram_max
            + radial_speed_max
            + tangential_speed_max
            + sdc_goal_angle_max
            + sdc_goal_distance_max
            + sdc_vel_x_max
            + sdc_vel_y_max
            + sdc_offroad_max,
            dtype=jnp.float32,
        )

        return specs.BoundedArray(
            shape=(total_dim,),
            minimum=min_bounds,
            maximum=max_bounds,
            dtype=jnp.float32,
        )
