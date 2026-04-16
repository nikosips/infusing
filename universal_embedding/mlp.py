from typing import Any, Optional

import flax
import flax.linen as nn
import jax.numpy as jnp



class Mlp(nn.Module):
  """MLP."""

  num_layers: int
  hidden_size_list: Optional[Any] = None
  dtype: jnp.dtype = jnp.float32
  skip_connect: bool=False

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,  # Shape: (N, ..., X)
  ) -> jnp.ndarray:  # Shape: (N, ..., E)
    y = x

    for i in range(self.num_layers):
      y = nn.Dense(self.hidden_size_list[i], dtype=self.dtype)(y)
      #dont use nonlinearity in the last layer
      if i == self.num_layers - 1:
        break
      y = nn.LayerNorm(dtype=self.dtype)(y)
      y = nn.gelu(y)

    if self.skip_connect and self.num_layers > 1: #skip connect doesn't make sense if there is no more than 1 layers == no non-linearity
      if y.shape[-1] != x.shape[-1]:
        x_transformed = nn.Dense(y.shape[-1], dtype=self.dtype)(x)

      else:
        x_transformed = x

      y = y + x_transformed

    return y