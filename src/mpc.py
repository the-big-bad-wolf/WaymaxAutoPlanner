import os
from typing import Tuple

import casadi
import jax
import numpy as np
import waymax.utils.geometry as utils
from casadi import external
from waymax import datatypes

from waymax_modified import create_road_circogram


def create_mpc_solver() -> casadi.Function:
    """
    Creates a compiled CasADi MPC solver for vehicle control with C code generation for speed.

    Returns:
        A compiled CasADi function that takes state and goal parameters and returns optimal control actions.
    """

    MAX_ACCEL = 6.0
    MAX_STEERING = 0.3
    MAX_SPEED = 20.0

    # Problem dimensions
    N = 20  # Prediction horizon
    dt = 0.1  # Time step

    # Create CasADi optimization variables
    opti = casadi.Opti()

    # State variables over the horizon: x, y, yaw, speed
    x = opti.variable(N + 1)
    y = opti.variable(N + 1)
    yaw = opti.variable(N + 1)
    speed = opti.variable(N + 1)

    # Control variables over the horizon: acceleration, steering
    accel = opti.variable(N)
    steering = opti.variable(N)

    # Parameters: initial state and target position
    params = opti.parameter(
        6
    )  # [start_x, start_y, start_yaw, start_speed, target_x, target_y]

    # Initial state constraints
    opti.subject_to(x[0] == params[0])
    opti.subject_to(y[0] == params[1])
    opti.subject_to(yaw[0] == params[2])
    opti.subject_to(speed[0] == params[3])

    # Target position
    target_x = params[4]
    target_y = params[5]

    # System dynamics (bicycle model)
    for i in range(N):
        opti.subject_to(x[i + 1] == x[i] + dt * speed[i] * casadi.cos(yaw[i]))
        opti.subject_to(y[i + 1] == y[i] + dt * speed[i] * casadi.sin(yaw[i]))
        opti.subject_to(
            yaw[i + 1] == yaw[i] + dt * speed[i] * steering[i] * MAX_STEERING
        )
        opti.subject_to(speed[i + 1] == speed[i] + dt * accel[i] * MAX_ACCEL)

        # Control constraints
        opti.subject_to(opti.bounded(-1, accel[i], 1))
        opti.subject_to(opti.bounded(-1, steering[i], 1))

        # State constraints
        opti.subject_to(opti.bounded(0.0, speed[i + 1], MAX_SPEED))

    # Add circogram parameter for obstacle distances
    circogram = opti.parameter(64)  # 64 ray measurements

    # Simplified objective function
    distance_to_goal = (x[N] - target_x) ** 2 + (y[N] - target_y) ** 2  # Terminal cost
    control_effort = casadi.sum1(accel**2 + 5.0 * steering**2)  # Control regularization

    opti.minimize(10 * distance_to_goal + control_effort)

    # Set up solver options
    p_opts = {"expand": True, "print_time": 0}
    s_opts = {
        "max_iter": 100,
        "print_level": 0,
        "warm_start_init_point": "yes",
        "acceptable_tol": 1e-2,
        "acceptable_obj_change_tol": 1e-2,
    }
    opti.solver("ipopt", p_opts, s_opts)

    # Create a CasADi function
    mpc_fn = opti.to_function(
        "mpc_solver",
        [params, circogram],
        [accel[0], steering[0]],
        ["params", "circogram"],
        ["optimal_accel", "optimal_steering"],
    )

    try:
        c_code_name = "mpc_solver"
        c_file_name = os.path.join(c_code_name + ".c")
        so_file_name = os.path.join(c_code_name + ".so")

        print(f"Generating C code at: {c_file_name}")

        code_gen = casadi.CodeGenerator(
            c_file_name, {"with_header": True, "with_mem": True}
        )
        code_gen.add(mpc_fn)
        code_gen.generate()

        # Check if file was created successfully
        if os.path.exists(c_file_name):
            print(f"C file created successfully at {c_file_name}")
            casadi_include = os.path.join(os.path.dirname(casadi.__file__), "include")
            compile_command = f"gcc -fPIC -shared -Ofast -march=native -I{casadi_include} {c_file_name} -o {so_file_name} -lipopt -ldl -lm"

            print(f"Running: {compile_command}")
            os.system(compile_command)

            # Load the compiled function
            mpc_fn = external("mpc_solver", so_file_name)
            print(f"Successfully compiled MPC solver to C.")

        else:
            print(f"Failed to create C file at {c_file_name}")
    except Exception as e:
        print(f"C compilation failed, using normal CasADi function: {str(e)}...")

    return mpc_fn


jit_select = jax.jit(datatypes.select_by_onehot, static_argnums=(2))
jit_observe_from_state = jax.jit(datatypes.sdc_observation_from_state)
jit_transform_points = jax.jit(utils.transform_points)
jit_create_road_circogram = jax.jit(create_road_circogram, static_argnums=(1))


def get_MPC_action(state: datatypes.SimulatorState) -> Tuple[float, float]:
    observation = jit_observe_from_state(state)
    sdc_trajectory = jit_select(
        observation.trajectory,
        observation.is_ego,
        keepdims=True,
    )
    sdc_velocity_xy = sdc_trajectory.vel_xy
    sdc_xy_goal = jit_select(
        state.log_trajectory.xy[..., -1, :],
        state.object_metadata.is_sdc,
        keepdims=True,
    )
    sdc_xy_goal = jit_transform_points(observation.pose2d.matrix, sdc_xy_goal)[0]

    start_x = 0.0
    start_y = 0.0
    start_yaw = 0.0
    start_vel_x = float(sdc_velocity_xy.flatten()[0])
    start_vel_y = float(sdc_velocity_xy.flatten()[1])
    start_speed = np.sqrt(start_vel_x**2 + start_vel_y**2)

    target_x = float(sdc_xy_goal[0])
    target_y = float(sdc_xy_goal[1])

    num_rays = 64
    road_circogram = jit_create_road_circogram(observation, num_rays)

    try:
        params = casadi.DM(
            [start_x, start_y, start_yaw, start_speed, target_x, target_y]
        )
        circogram = casadi.DM(road_circogram)

        # Call the function correctly
        compiled_mpc_solver = create_mpc_solver()
        result = compiled_mpc_solver(params, circogram)

        # Extract results (must convert to scalar values)
        optimal_accel = float(result[0])
        optimal_steering = float(result[1])

    except Exception as e:
        print(f"MPC solver failed: {str(e)[:100]}...")
        # Fallback to a simple controller
        optimal_steering = 0.0  # No steering
        optimal_accel = 0.0  # No acceleration

    return (optimal_accel, optimal_steering)
