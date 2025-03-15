import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.jax import SequentialTrainer
from waymax import agents
from waymax import config as _config
from waymax import dataloader, dynamics, metrics

from waymax_modified import WaymaxEnv
from waymax_wrapper import WaymaxWrapper

# Set the backend to "jax" or "numpy"
config.jax.backend = "jax"
config.jax.device = jax.devices("cpu")[0]


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
    data_iter = dataloader.simulator_state_generator(config=data_loader_config)
    sim_agent_config = _config.SimAgentConfig(
        agent_type=_config.SimAgentType.IDM,
        controlled_objects=_config.ObjectType.NON_SDC,
    )
    print("Available metrics:", metrics.get_metric_names())
    metrics_config = _config.MetricsConfig(
        metrics_to_run=("sdc_progression", "offroad")
    )
    reward_config = _config.LinearCombinationRewardConfig(
        rewards={"sdc_progression": 1.0, "offroad": -1.0},
    )
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        metrics=metrics_config,
        rewards=reward_config,
        max_num_objects=max_num_objects,
        sim_agents=[sim_agent_config],
    )
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
    env = WaymaxEnv(
        dynamics_model=dynamics_model,
        config=env_config,
        sim_agent_actors=[agents.create_sim_agents_from_config(sim_agent_config)],
        sim_agent_params=[{}],
    )
    return env, data_iter


class MLP_Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(32)(inputs["states"]))
        x = nn.relu(nn.Dense(32)(x))
        x = nn.Dense(self.num_actions)(x)  # type: ignore
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return nn.tanh(x), log_std, {}


class MLP_Value(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device=None, clip_actions=False, **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(16)(inputs["states"]))
        x = nn.relu(nn.Dense(16)(x))
        x = nn.Dense(1)(x)
        return x, {}


class CNN_Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        # Split inputs - extract first 5 features and the rest for CNN
        first_five = inputs["states"][:, :5]  # First 5 features
        cnn_input = inputs["states"][:, 5:]

        # Reshape CNN input for 1D convolution - add channel dimension
        batch_size = cnn_input.shape[0]
        cnn_input = jnp.reshape(
            cnn_input, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply 3 convolutional layers with circular padding
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(cnn_input)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.relu(x)

        # Flatten output and concatenate with first_five features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([first_five, x], axis=1)  # Combine with first 5 features

        # Final MLP layers
        x = nn.relu(nn.Dense(64)(x))
        x = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))

        return nn.tanh(x), log_std, {}


class CNN_Value(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device=None, clip_actions=False, **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact
    def __call__(self, inputs, role):
        # Split inputs - extract first 5 features and the rest for CNN
        first_five = inputs["states"][:, :5]  # First 5 features
        cnn_input = inputs["states"][:, 5:]

        # Reshape CNN input for 1D convolution - add channel dimension
        batch_size = cnn_input.shape[0]
        cnn_input = jnp.reshape(
            cnn_input, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply convolutional layers with circular padding
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(cnn_input)
        x = nn.relu(x)

        # Flatten output and concatenate with first_five features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([first_five, x], axis=1)  # Combine with first 5 features

        # Final MLP layers
        x = nn.relu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)

        return x, {}


if __name__ == "__main__":
    env, data_iter = setup_waymax()
    env = WaymaxWrapper(env, data_iter)

    # instantiate a memory as rollout buffer (any memory can be used for this)
    mem_size = 8192
    memory = RandomMemory(memory_size=mem_size, num_envs=1)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = CNN_Policy(env.observation_space, env.action_space)
    models["value"] = CNN_Value(env.observation_space, env.action_space)

    # instantiate models' state dict
    for role, model in models.items():
        model.init_state_dict(role)

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = mem_size  # memory_size
    cfg["mini_batches"] = 512
    cfg["random_timesteps"] = 0
    cfg["entropy_loss_scale"] = 0
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()

    # load the latest checkpoint (adjust the path as needed)
    # agent.load("runs/25-03-10_14-13-28-381661_PPO/checkpoints/best_agent.pickle")
    # visualize the training
    trainer.timesteps = 1000
    trainer.headless = False
    trainer.eval()
