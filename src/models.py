from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
import flax.linen as nn
from jax import numpy as jnp


class Policy_Model(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=True,
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
        num_rays = 64
        circogram = inputs["states"][:, 0:num_rays]
        circogram_radial_speed = inputs["states"][:, num_rays : 2 * num_rays]
        circogram_tangential_speed = inputs["states"][:, 2 * num_rays : 3 * num_rays]
        misc_features = inputs["states"][:, 3 * num_rays :]

        # Reshape each circogram component to add a channel dimension
        batch_size = inputs["states"].shape[0]
        circogram = jnp.reshape(circogram, (batch_size, num_rays, 1))
        circogram_radial_speed = jnp.reshape(
            circogram_radial_speed, (batch_size, num_rays, 1)
        )
        circogram_tangential_speed = jnp.reshape(
            circogram_tangential_speed, (batch_size, num_rays, 1)
        )

        # Shape becomes [batch, num_rays, 3]
        circogram_combined = jnp.concatenate(
            [circogram, circogram_radial_speed, circogram_tangential_speed], axis=2
        )

        # Apply 3 convolutional layers with circular padding
        x = nn.Conv(features=64, kernel_size=15, padding="CIRCULAR")(circogram_combined)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=128, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)

        # Flatten output and concatenate with remaining features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([x, misc_features], axis=1)

        # Final MLP layers
        x = nn.leaky_relu(nn.Dense(64)(x))
        x = nn.leaky_relu(nn.Dense(64)(x))
        x = nn.Dense(self.num_actions)(x)  # type: ignore
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))

        # Apply tanh to the output for bicycle action
        x = nn.tanh(x)

        # Transform output to match trajectory sampling
        # Split the output into three parts
        # x1 = x[:, :8]  # First 8 elements
        # x2 = x[:, 8:16]  # Next 8 elements
        # x3 = x[:, 16:]  # The rest

        # Transform first 8 elements to be between -1 and 1 using tanh
        # x1 = nn.tanh(x1)

        # Transform next 8 elements to be between 1e-5 and 0.4
        # Apply sigmoid and scale
        # x2 = 1e-5 + nn.sigmoid(x2) * (0.4 - 1e-5)

        # Transform the rest to be between -0.2 and 0.2
        # x3 = nn.tanh(x3) * 0.2

        # Combine the transformed parts back together
        # x = jnp.concatenate([x1, x2, x3], axis=1)

        return x, log_std, {}


class Value_Model(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device=None, clip_actions=False, **kwargs
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact
    def __call__(self, inputs, role):
        num_rays = 64
        circogram = inputs["states"][:, 0:num_rays]
        circogram_radial_speed = inputs["states"][:, num_rays : 2 * num_rays]
        circogram_tangential_speed = inputs["states"][:, 2 * num_rays : 3 * num_rays]
        misc_features = inputs["states"][:, 3 * num_rays :]

        # Reshape each circogram component to add a channel dimension
        batch_size = inputs["states"].shape[0]
        circogram = jnp.reshape(circogram, (batch_size, num_rays, 1))
        circogram_radial_speed = jnp.reshape(
            circogram_radial_speed, (batch_size, num_rays, 1)
        )
        circogram_tangential_speed = jnp.reshape(
            circogram_tangential_speed, (batch_size, num_rays, 1)
        )

        # Shape becomes [batch, num_rays, 3]
        circogram_combined = jnp.concatenate(
            [circogram, circogram_radial_speed, circogram_tangential_speed], axis=2
        )

        # Apply 3 convolutional layers with circular padding
        x = nn.Conv(features=64, kernel_size=15, padding="CIRCULAR")(circogram_combined)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=128, kernel_size=3, padding="CIRCULAR")(x)
        x = nn.leaky_relu(x)

        # Flatten output and concatenate with remaining features
        x = x.reshape(batch_size, -1)  # Flatten conv output
        x = jnp.concatenate([x, misc_features], axis=1)

        # Final MLP layers
        x = nn.leaky_relu(nn.Dense(64)(x))
        x = nn.leaky_relu(nn.Dense(64)(x))
        x = nn.Dense(1)(x)

        return x, {}
