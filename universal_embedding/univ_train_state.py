from typing import Any, Dict, Optional
from flax import struct
import optax
import jax.numpy as jnp



@struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(
      default=None, pytree_node=False
  )
  opt_state: Optional[optax.OptState] = None
  params: Optional[Any] = struct.field(default_factory=dict)
  pretrained_params: Optional[Any] = struct.field(default_factory=dict)
  global_step: Optional[int] = 0
  model_state: Optional[Any] = struct.field(default_factory=dict)
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None
  # NOTE: When using the raw TrainState as the target for checkpoint restoration
  #  in Flax, you should provide the pytree structure, otherwise it might just
  #  silenty ignore restoring the checkpoint subtree if you use with an empty
  #  dict when setting `allow_partial_mpa_restoration=True` and if you set it
  #  to None (e.g., for `metadata`` above), Flax replaces it with a state dict.

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default