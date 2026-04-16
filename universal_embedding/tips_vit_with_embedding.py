"""TIPS Vision Transformer with an extra embedding head and cosine-style classifier."""

from typing import Any, Callable, Iterable

import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections

from tips.scenic.models.tips import VisionEncoder
from tips.scenic.utils import checkpoint

from universal_embedding import model
from universal_embedding.embedding_head import EmbeddingHead


Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
JAX_PRECISION = "default"


def _safe_l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    """Applies L2 normalization with numerical protection."""
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return x / norm


class TIPSViTWithEmbedding(nn.Module):
    """TIPS vision encoder followed by the universal embedding head."""

    num_classes: int
    output_dim: int = -1
    dataset_meta_data: Any = None
    config: ml_collections.ConfigDict = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        domain: int,
        *,
        train: bool,
        init: bool = False,
        debug: bool = False,
        return_feats: bool = False,
        project_feats: bool = True,
        domain_agnostic: bool = False,
    ):
        """Runs the TIPS encoder and the downstream embedding/classification head.

        Args:
            x: Input images of shape [batch, height, width, channels].
            domain: Domain index for the current batch.
            train: Whether the model is in training mode.
            init: Whether this call is part of parameter initialization.
            debug: Forwarded to the TIPS encoder / embedding head.
            return_feats: If True, skip classifier computation in the embedding head.
            project_feats: Currently unused, kept for interface compatibility.
            domain_agnostic: Forwarded to the embedding head.

        Returns:
            A dictionary containing embeddings and optional classifier outputs.
        """
        del project_feats

        outputs = {"embeddings": {}}

        if train or init:
            outputs["classifier"] = {}

        tips_encoder = VisionEncoder(
            variant="B/14",
            pooling="tok",
            num_cls_tokens=2,
        )

        _, embeddings_vision = tips_encoder(
            x,
            train=train,
            debug=debug,
        )

        # Use the first CLS token as the main image embedding.
        x = embeddings_vision[:, 0, :]

        # Normalize the backbone output before passing it to the embedding head.
        x = _safe_l2_normalize(x, axis=1)
        outputs["embeddings"]["backbone_out_embedd"] = x

        outputs = EmbeddingHead(
            output_dim=self.output_dim,
            dataset_meta_data=self.dataset_meta_data,
            config=self.config,
            name="Embedding_Head",
        )(
            outputs,
            domain,
            train=train,
            debug=debug,
            init=init,
            return_feats=return_feats,
            domain_agnostic=domain_agnostic,
        )

        return outputs


class ViTWithEmbeddingModel(model.UniversalEmbeddingModel):
    """Scenic wrapper for the TIPS ViT + embedding-head model."""

    def build_flax_model(self) -> nn.Module:
        dtype_str = self.config.get("model_dtype_str", "float32")

        if dtype_str != "float32":
            raise ValueError(
                "`dtype` is not propagated correctly in the current implementation, "
                "so only `float32` is supported for now."
            )

        return TIPSViTWithEmbedding(
            num_classes=self.dataset_meta_data["num_classes"],
            dataset_meta_data=self.dataset_meta_data,
            config=self.config,
            output_dim=self.config.model.output_dim,
        )

    def default_flax_model_config(self) -> ml_collections.ConfigDict:
        return None

    def load_tips_params(
        self,
        train_state: Any,
        params_path: str,
    ) -> Any:
        """Loads a TIPS checkpoint into the VisionEncoder subtree."""
        restored_params = checkpoint.load_checkpoint(
            params_path,
            train_state.params["VisionEncoder_0"],
        )

        updated_params = flax.core.unfreeze(train_state.params)
        updated_params["VisionEncoder_0"].update(restored_params)
        updated_params = flax.core.freeze(updated_params)

        return train_state.replace(params=updated_params)
