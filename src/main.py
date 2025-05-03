import dataclasses

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


def setup_waymax():
    path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
    path = "data/training_tfexample.tfrecord@5"
    max_num_objects = 128
    data_loader_config = dataclasses.replace(
        _config.WOD_1_1_0_TRAINING,
        path=path,
        max_num_objects=max_num_objects,
        max_num_rg_points=30000,
    )
    scenario_loader = dataloader.simulator_state_generator(config=data_loader_config)
    IDM_agent_config = _config.SimAgentConfig(
        agent_type=_config.SimAgentType.IDM,
        controlled_objects=_config.ObjectType.NON_SDC,
    )
    IDM_agent = agents.create_sim_agents_from_config(
        config=IDM_agent_config,
    )
    metrics_config = _config.MetricsConfig(
        metrics_to_run=("sdc_progression", "offroad", "overlap")
    )
    reward_config = _config.LinearCombinationRewardConfig(
        rewards={"sdc_progression": 1.0, "offroad": -20.0, "overlap": -20.0},
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
        sim_agent_actors=[],
        sim_agent_params=[],
    )
    return env, scenario_loader


if __name__ == "__main__":
    # Set the backend to "jax" or "numpy"
    config.jax.backend = "numpy"

    env, scenario_loader = setup_waymax()
    env = WaymaxWrapper(env, scenario_loader, action_space_type="bicycle", MPC=False)

    # instantiate a memory as rollout buffer
    mem_size = 16384
    memory = RandomMemory(memory_size=mem_size, num_envs=1)

    models = {}
    models["policy"] = Policy_Model(env.observation_space, env.action_space)
    models["value"] = Value_Model(env.observation_space, env.action_space)
    for role, model in models.items():
        model.init_state_dict(role)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = mem_size  # memory_size
    cfg["mini_batches"] = 128
    cfg["learning_epochs"] = 8
    cfg["entropy_loss_scale"] = 0.01
    # cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1}

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # load the latest checkpoint (adjust the path as needed)
    # agent.load("runs/25-05-02_03-22-44-582590_PPO/checkpoints/best_agent.pickle")

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()

    # visualize the agent
    trainer.timesteps = 1000
    trainer.headless = False
    trainer.eval()
