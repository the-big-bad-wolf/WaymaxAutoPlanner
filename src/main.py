import dataclasses
import datetime

from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.jax import RandomMemory
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.trainers.jax import SequentialTrainer
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
MODE = "train"  # "train" or "eval"

# Agent configuration
ACTION_SPACE_TYPE = "bicycle"  # bicycle/bicycle_mpc/trajectory_sampling
MODEL_PATH = "runs/25-06-04_01-24_bicycle/checkpoints/best_agent.pickle"

# Training and evaluation parameters
TRAIN_TIMESTEPS = 3000000
EVAL_TIMESTEPS = 1000
HEADLESS = True  # Set to False to enable visualization
#####################################################################


def setup_waymax(data_path: str, use_idm: bool = False):
    # Existing setup_waymax code unchanged
    max_num_objects = 128
    data_loader_config = dataclasses.replace(
        _config.WOD_1_1_0_TRAINING,
        path=data_path,
        max_num_objects=max_num_objects,
        max_num_rg_points=30000,
    )
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
        rewards={"sdc_progression": 1.0, "offroad": -1.0, "overlap": -2.0},
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


def setup_agent(env):
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
    ppo_cfg["mini_batches"] = 32
    ppo_cfg["learning_epochs"] = 8
    ppo_cfg["entropy_loss_scale"] = 0.01
    ppo_cfg["ratio_clip"] = 0.2
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
    ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.05}
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1}
    ppo_cfg["experiment"]["experiment_name"] = "{}_{}".format(
        datetime.datetime.now().strftime("%y-%m-%d_%H-%M"), ACTION_SPACE_TYPE
    )

    ppo_agent = PPO(
        models=ppo_models,
        memory=ppo_memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    return ppo_agent, ppo_cfg


if __name__ == "__main__":
    # Set the backend to "jax" or "numpy"
    config.jax.backend = "numpy"

    if MODE == "train":
        print("Starting TRAINING mode...")
        # Setup training environment
        training_path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
        env, scenario_loader = setup_waymax(training_path)
        env = WaymaxWrapper(env, scenario_loader, action_space_type=ACTION_SPACE_TYPE)

        # Setup agent
        ppo_agent, ppo_cfg = setup_agent(env)

        # Load model if continuing training
        if MODEL_PATH:
            print(f"Loading existing model from {MODEL_PATH}")
            ppo_agent.load(MODEL_PATH)

        # Configure trainer using parameters from config section
        cfg_trainer = {
            "timesteps": TRAIN_TIMESTEPS,
            "headless": HEADLESS,
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
        test_path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/testing/testing_tfexample.tfrecord@150"
        env, scenario_loader = setup_waymax(test_path, use_idm=True)
        env = WaymaxWrapper(env, scenario_loader, action_space_type=ACTION_SPACE_TYPE)

        # Setup agent with the evaluation environment
        ppo_agent, ppo_cfg = setup_agent(env)

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
        trainer.eval()

        # Print evaluation statistics
        print("Evaluation complete!")
        print("Episode statistics:", env.get_episode_statistics())
