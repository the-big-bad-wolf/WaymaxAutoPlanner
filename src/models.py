from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
import flax.linen as nn
from jax import numpy as jnp


class Policy_Model(GaussianMixin, Model):
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
        road_circogram = inputs["states"][:, 0:64]
        object_circogram = inputs["states"][:, 64:128]
        misc_features = inputs["states"][:, 128:]

        # Reshape road_circogram for 1D convolution - add channel dimension
        batch_size = inputs["states"].shape[0]
        road_circogram = jnp.reshape(
            road_circogram, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply 5 convolutional layers with circular padding
        x = nn.Conv(features=8, kernel_size=5, padding="CIRCULAR")(road_circogram)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)

        # Reshape object_circogram for 1D convolution - add channel dimension
        object_circogram = jnp.reshape(
            object_circogram, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply 5 convolutional layers with circular padding
        x2 = nn.Conv(features=8, kernel_size=5, padding="CIRCULAR")(object_circogram)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)

        # Concatenate the outputs of the two branches
        x = jnp.concatenate([x, x2], axis=1)

        # Flatten output and concatenate with remaining features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([x, misc_features], axis=1)

        # Final MLP layers
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.Dense(self.num_actions)(x)  # type: ignore
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))

        # Make sure the diagonal elements of the cholesky factor are non-negative (it is neccessary to also clip the outputted params as the the standard deviation in the training process may push them negative)
        x = jnp.concatenate([x[:, :-8], nn.softplus(x[:, -8:])], axis=1)
        return x, log_std, {}


class Value_Model(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device=None, clip_actions=False, **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact
    def __call__(self, inputs, role):
        # Split inputs similar to Policy_Model
        road_circogram = inputs["states"][:, 0:64]
        object_circogram = inputs["states"][:, 64:128]
        misc_features = inputs["states"][:, 128:]

        # Reshape road_circogram for 1D convolution - add channel dimension
        batch_size = inputs["states"].shape[0]
        road_circogram = jnp.reshape(
            road_circogram, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply 5 convolutional layers with circular padding
        x = nn.Conv(features=8, kernel_size=5, padding="CIRCULAR")(road_circogram)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)

        # Reshape object_circogram for 1D convolution - add channel dimension
        object_circogram = jnp.reshape(
            object_circogram, (batch_size, -1, 1)
        )  # [batch, features, channels]

        # Apply 5 convolutional layers with circular padding
        x2 = nn.Conv(features=8, kernel_size=5, padding="CIRCULAR")(object_circogram)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=32, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)
        x2 = nn.Conv(features=16, kernel_size=3, padding="CIRCULAR")(x2)
        x2 = nn.leaky_relu(x2)

        # Concatenate the outputs of the two branches
        x = jnp.concatenate([x, x2], axis=1)

        # Flatten output and concatenate with remaining features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([x, misc_features], axis=1)

        # Final MLP layers
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.leaky_relu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)

        return x, {}
