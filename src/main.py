import dataclasses

from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.jax import RandomMemory
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.trainers.jax import SequentialTrainer
from waymax import agents
from waymax import config as _config
from waymax import dataloader, dynamics

from models import Policy_Model, Value_Model, Critic_Model
from waymax_modified import WaymaxEnv
from waymax_wrapper import WaymaxWrapper


def setup_waymax():
    path = "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
    # path = "data/training_tfexample.tfrecord@5"
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
        sim_agent_actors=[],
        sim_agent_params=[],
    )
    return env, scenario_loader


if __name__ == "__main__":
    # Set the backend to "jax" or "numpy"
    config.jax.backend = "numpy"

    env, scenario_loader = setup_waymax()
    env = WaymaxWrapper(
        env, scenario_loader, action_space_type="trajectory_sampling", MPC=False
    )

    # instantiate a memory as rollout buffer
    mem_size = 16384
    ppo_memory = RandomMemory(memory_size=mem_size, num_envs=1)

    ppo_models = {}
    ppo_models["policy"] = Policy_Model(env.observation_space, env.action_space)
    ppo_models["value"] = Value_Model(env.observation_space, env.action_space)
    for role, model in ppo_models.items():
        model.init_state_dict(role)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = mem_size  # memory_size
    ppo_cfg["mini_batches"] = 32
    ppo_cfg["learning_epochs"] = 8
    ppo_cfg["entropy_loss_scale"] = 0.005
    ppo_cfg["ratio_clip"] = 0.2
    # cfg["learning_rate"] = 1e-4
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
    ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1}

    ppo_agent = PPO(
        models=ppo_models,
        memory=ppo_memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    sac_memory = RandomMemory(memory_size=1000000, num_envs=1)
    sac_models = {}
    sac_models["policy"] = Policy_Model(env.observation_space, env.action_space)
    sac_models["critic_1"] = Critic_Model(env.observation_space, env.action_space)
    sac_models["critic_2"] = Critic_Model(env.observation_space, env.action_space)
    sac_models["target_critic_1"] = Critic_Model(
        env.observation_space, env.action_space
    )
    sac_models["target_critic_2"] = Critic_Model(
        env.observation_space, env.action_space
    )

    for role, model in sac_models.items():
        model.init_state_dict(role)

    sac_cfg = SAC_DEFAULT_CONFIG.copy()
    sac_cfg["batch_size"] = 512
    sac_cfg["learning_starts"] = 10000
    sac_cfg["random_steps"] = 5000
    sac_cfg["actor_learning_rate"] = 1e-4
    sac_cfg["critic_learning_rate"] = 1e-4
    sac_cfg["entropy_learning_rate"] = 1e-4
    sac_cfg["state_preprocessor"] = RunningStandardScaler
    sac_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}

    sac_agent = SAC(
        models=sac_models,
        memory=sac_memory,
        cfg=sac_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # load the latest checkpoint (adjust the path as needed)
    # sac_agent.load("runs/25-05-10_01-02-59-821787_SAC/checkpoints/agent_500000.pickle")

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[ppo_agent])

    # start training
    trainer.train()

    # visualize the agent
    trainer.timesteps = 1000
    trainer.headless = False
    trainer.eval()
