import dataclasses
import datetime
import math
import os
from typing import Sequence, Tuple

from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.jax import RandomMemory
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.trainers.jax import SequentialTrainer


def generate_sharded_filenames_with_range(path: str) -> Sequence[str]:
    """Enhanced version that supports @start-end format"""
    base_name, shard_spec = path.split("@")

    if "-" in shard_spec:
        start_shard, end_shard = map(int, shard_spec.split("-"))
        total_shards = 1000  # Waymo training dataset total
        shard_range = range(start_shard, end_shard + 1)
    else:
        total_shards = int(shard_spec)
        shard_range = range(total_shards)

    shard_width = max(5, int(math.log10(total_shards) + 1))
    format_str = base_name + "-%0" + str(shard_width) + "d-of-%05d"
    return [format_str % (i, total_shards) for i in shard_range]


# Monkey patch the function
from waymax.dataloader import dataloader_utils

dataloader_utils.generate_sharded_filenames = generate_sharded_filenames_with_range


from waymax import agents
from waymax import config as _config
from waymax import dataloader, dynamics

from models import Policy_Model, Value_Model
from waymax_modified import WaymaxEnv
from waymax_wrapper import WaymaxWrapper

#####################################################################
# CONFIGURATION - Edit these variables to change behavior
#####################################################################
# Run mode
MODE = "eval"  # "train" or "eval"

# Agent configuration
ACTION_SPACE_TYPE = "bicycle"  # bicycle/bicycle_mpc/trajectory_sampling
MODEL_PATH = "runs/25-06-07_18-29_bicycle/training/25-06-07_18-29_bicycle/checkpoints/best_agent.pickle"

# Training and evaluation parameters
TRAIN_TIMESTEPS = 5000000
EVAL_TIMESTEPS = 1000
HEADLESS = False  # Set to False to enable visualization

# Dataset configuration
TRAIN_SHARD_START = 0
TRAIN_SHARD_COUNT = 850  # Use first 850 shards for training
EVAL_SHARD_START = 850  # Use remaining shards for evaluation
EVAL_SHARD_COUNT = 150  # Use last 150 shards for evaluation
#####################################################################


def setup_waymax(
    data_path: str, use_idm: bool = False, shard_start: int = 0, shard_count: int = 1000
):
    max_num_objects = 128

    # Create shard-specific path with correct format
    if "@" in data_path:
        base_path = data_path.split("@")[0]
        # Use the correct format: @start-end
        shard_end = shard_start + shard_count - 1
        shard_path = f"{base_path}@{shard_start}-{shard_end}"
    else:
        shard_end = shard_start + shard_count - 1
        shard_path = f"{data_path}@{shard_start}-{shard_end}"

    data_loader_config = dataclasses.replace(
        _config.WOD_1_1_0_TRAINING,
        path=shard_path,
        max_num_objects=max_num_objects,
        max_num_rg_points=30000,
    )

    print(f"Loading data from: {shard_path}")
    scenario_loader = dataloader.simulator_state_generator(config=data_loader_config)

    # Set up sim agents based on the use_idm parameter
    sim_agent_actors = []
    sim_agent_params = []

    if use_idm:
        # Create and add IDM agent when requested
        IDM_agent_config = _config.SimAgentConfig(
            agent_type=_config.SimAgentType.IDM,
            controlled_objects=_config.ObjectType.NON_SDC,
        )
        IDM_agent = agents.create_sim_agents_from_config(
            config=IDM_agent_config,
        )
        sim_agent_actors = [IDM_agent]
        sim_agent_params = [{}]  # Default empty params for the IDM agent

    metrics_config = _config.MetricsConfig(
        metrics_to_run=("sdc_progression", "offroad", "overlap")
    )
    reward_config = _config.LinearCombinationRewardConfig(
        rewards={"sdc_progression": 1.0, "offroad": -2.0, "overlap": -4.0},
    )
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        metrics=metrics_config,
        rewards=reward_config,
        max_num_objects=max_num_objects,
    )
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)

    env = WaymaxEnv(
        dynamics_model=dynamics_model,
        config=env_config,
        sim_agent_actors=sim_agent_actors,
        sim_agent_params=sim_agent_params,
    )
    return env, scenario_loader


def setup_agent(env, experiment_dir=None, experiment_name=None, is_eval=False):
    """Create and configure the PPO agent"""
    mem_size = 16384
    ppo_memory = RandomMemory(memory_size=mem_size, num_envs=1)

    ppo_models = {}
    ppo_models["policy"] = Policy_Model(env.observation_space, env.action_space)
    ppo_models["value"] = Value_Model(env.observation_space, env.action_space)
    for role, model in ppo_models.items():
        model.init_state_dict(role)

    if ACTION_SPACE_TYPE == "bicycle_mpc":
        ppo_models["policy"].init_parameters(method_name="zeros")

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = mem_size
    ppo_cfg["mini_batches"] = 128
    ppo_cfg["learning_epochs"] = 20
    ppo_cfg["entropy_loss_scale"] = 1.0
    ppo_cfg["ratio_clip"] = 0.2
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
    ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.15}
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1}

    if experiment_name is None:
        experiment_name = "{}_{}".format(
            datetime.datetime.now().strftime("%y-%m-%d_%H-%M"), ACTION_SPACE_TYPE
        )

    ppo_cfg["experiment"]["experiment_name"] = experiment_name

    if experiment_dir is not None:
        ppo_cfg["experiment"]["directory"] = experiment_dir

    # Disable checkpoints during evaluation
    if is_eval:
        ppo_cfg["experiment"]["checkpoint_interval"] = 0

    ppo_agent = PPO(
        models=ppo_models,
        memory=ppo_memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    return ppo_agent


def create_experiment_directories(base_name: str) -> Tuple[str, str]:
    """Create organized experiment directories"""
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    experiment_name = f"{timestamp}_{base_name}"

    base_dir = os.path.join("runs", experiment_name)
    train_dir = os.path.join(base_dir, "training")
    eval_dir = os.path.join(base_dir, "evaluation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    return train_dir, eval_dir


if __name__ == "__main__":
    # Set the backend to "jax" or "numpy"
    config.jax.backend = "numpy"

    if MODE == "train":
        print("Starting TRAINING mode...")

        # Create organized directories
        train_dir, eval_dir = create_experiment_directories(ACTION_SPACE_TYPE)

        # Setup training environment
        training_path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord"
        env, scenario_loader = setup_waymax(
            training_path,
            use_idm=False,
            shard_start=TRAIN_SHARD_START,
            shard_count=TRAIN_SHARD_COUNT,
        )

        # Initialize wrapper with training directory and video settings
        env = WaymaxWrapper(
            env,
            scenario_loader,
            action_space_type=ACTION_SPACE_TYPE,
            save_dir=train_dir,
            save_videos=False,  # Disable videos during training for speed
        )

        # Setup agent with training directory
        ppo_agent = setup_agent(env, experiment_dir=train_dir)

        # Load model if continuing training
        if MODEL_PATH:
            print(f"Loading existing model from {MODEL_PATH}")
            ppo_agent.load(MODEL_PATH)

        # Configure trainer using parameters from config section
        cfg_trainer = {
            "timesteps": TRAIN_TIMESTEPS,
            "headless": False,
        }
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[ppo_agent])

        # Start training
        print(f"Training for {cfg_trainer['timesteps']} timesteps...")
        trainer.train()

        # Print final statistics
        print("Training complete!")
        print("Episode statistics:", env.get_episode_statistics())

    else:  # Evaluation mode
        print("Starting EVALUATION mode...")
        # Setup evaluation environment directly
        training_path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord"
        env, scenario_loader = setup_waymax(
            training_path,
            use_idm=True,
            shard_start=EVAL_SHARD_START,
            shard_count=EVAL_SHARD_COUNT,
        )

        # Extract the training experiment folder from the model path and configure eval logging
        model_dir = os.path.dirname(os.path.dirname(MODEL_PATH))
        eval_dir = os.path.join(model_dir, "evaluation")

        env = WaymaxWrapper(
            env,
            scenario_loader,
            action_space_type=ACTION_SPACE_TYPE,
            save_dir=eval_dir,
            save_videos=HEADLESS,
        )

        # Setup agent with the evaluation environment - disable checkpoints
        ppo_agent = setup_agent(
            env, experiment_dir=eval_dir, experiment_name="evaluation", is_eval=True
        )

        # Load saved model
        print(f"Loading model from {MODEL_PATH}")
        ppo_agent.load(MODEL_PATH)

        # Configure trainer for evaluation using parameters from config section
        cfg_trainer = {
            "timesteps": EVAL_TIMESTEPS,
            "headless": HEADLESS,
        }
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[ppo_agent])

        # Start evaluation
        print(f"Evaluating for {cfg_trainer['timesteps']} timesteps...")
        print(f"Evaluation logs will be saved to: {eval_dir}")
        trainer.eval()

        # Print evaluation statistics
        print("Evaluation complete!")
        print("Episode statistics:", env.get_episode_statistics())
