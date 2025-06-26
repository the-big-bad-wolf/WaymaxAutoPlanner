from typing import Tuple

import jax
import jax.numpy as jnp
from waymax import datatypes

MAX_CIRCOGRAM_DIST = 60.0  # Maximum distance for the circogram (meters)
Z_TOLERANCE = 3.0  # Filter out objects and road graph points that are on a different height (meters)


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
