from typing import Any, Tuple, override

import casadi
import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import waymax.utils.geometry as utils
from dm_env import specs
from tqdm import tqdm
from waymax import agents
from waymax import config as _config
from waymax import dataloader, datatypes, dynamics
from waymax import env as _env
from waymax import visualization
from waymax.env import typedefs as types
from waymax.metrics.roadgraph import is_offroad
import os
from casadi import external


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
