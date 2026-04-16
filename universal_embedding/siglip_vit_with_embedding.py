"""Vision Transformer with an extra projection head and cosine-style classifier."""

from typing import Any, Callable, Iterable

import logging

import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections

import big_vision.models.vit as big_vision_vit
import big_vision.utils as u

from universal_embedding import model
from universal_embedding.embedding_head import EmbeddingHead


Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
JAX_PRECISION = "default"


def _safe_l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return x / norm


class BigVisionViTWithEmbedding(big_vision_vit._Model):
    """Big Vision ViT backbone with an additional embedding/classification head."""

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
        """Applies the ViT backbone and the universal embedding head.

        Args:
            x: Input images of shape [batch, height, width, channels].
            domain: Domain index for the current batch.
            train: Whether the model is in training mode.
            init: Whether this call is happening during parameter initialization.
            debug: Forwarded to the embedding head.
            return_feats: If True, skip classifier computations in the head.
            project_feats: Currently unused, kept for interface compatibility.
            domain_agnostic: Forwarded to the embedding head.

        Returns:
            A dictionary containing embeddings and optional classifier outputs.
        """
        del project_feats

        outputs = {"embeddings": {}}
        if train or init:
            outputs["classifier"] = {}
            
        image = jnp.asarray(x, self.dtype_mm)

        # Patch extraction.
        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=self.dtype_mm,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add positional embeddings before adding any extra token.
        x = x + big_vision_vit.get_posemb(
            self,
            self.posemb,
            (h, w),
            c,
            "pos_embedding",
            x.dtype,
        )

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x, _ = big_vision_vit.Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)

        encoded = x

        if self.pool_type == "map":
            x = big_vision_vit.MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
            )(x)
        elif self.pool_type == "gap":
            x = jnp.mean(x, axis=1)
        elif self.pool_type in {"0", "tok"}:
            x = x[:, 0]
            if self.pool_type == "tok":
                encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        # Normalize the backbone embedding before handing it to the embedding head.
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
    """Scenic wrapper for the ViT + embedding-head model."""

    def build_flax_model(self) -> nn.Module:
        dtype_str = self.config.get("model_dtype_str", "float32")

        if dtype_str != "float32":
            raise ValueError(
                "`dtype` is not propagated correctly in the current implementation, "
                "so only `float32` is supported for now."
            )

        return BigVisionViTWithEmbedding(
            num_classes=self.dataset_meta_data["num_classes"],
            dataset_meta_data=self.dataset_meta_data,
            config=self.config,
            mlp_dim=self.config.model.mlp_dim,
            depth=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            posemb="learn",
            rep_size=self.config.model.representation_size,
            patch_size=self.config.model.patches.size,
            width=self.config.model.hidden_size,
            pool_type="map",
            dropout=self.config.model.get("dropout_rate"),
            dtype_mm=getattr(jnp, dtype_str),
            output_dim=self.config.model.output_dim,
        )

    def default_flax_model_config(self) -> ml_collections.ConfigDict:
        return None

    def load_siglip_params(
        self,
        train_state: Any,
        params_path: str,
        model_cfg: ml_collections.ConfigDict,
        params_not_to_load: Any,
    ) -> Any:
        restored_params = load(
            train_state.params,
            params_path,
            model_cfg,
            dont_load=params_not_to_load,
        )
        return train_state.replace(params=flax.core.freeze(restored_params))


def load(init_params, init_file, model_cfg, dont_load=()):
    """Loads a Big Vision checkpoint into the current model parameter structure.

    This helper:
    - loads the checkpoint,
    - converts old checkpoint layouts if needed,
    - handles scan/non-scan conversion,
    - merges checkpoint params into the current initialized params,
    - resamples positional embeddings when needed.
    """
    restored_params = u.load_params(init_file)
    restored_params = big_vision_vit.fix_old_checkpoints(restored_params)

    # Convert between scan and non-scan layouts if needed.
    if model_cfg.get("scan") and "encoderblock" not in restored_params["Transformer"]:
        restored_params = big_vision_vit.pyloop_to_scan(restored_params)
    if (not model_cfg.get("scan")) and "encoderblock" in restored_params["Transformer"]:
        restored_params = big_vision_vit.scan_to_pyloop(restored_params)

    restored_params = merge_params(restored_params, init_params, dont_load)

    # Resample positional embeddings if the initialized model expects a different
    # resolution or sequence length.
    if init_params and "pos_embedding" in init_params:
        restored_params["pos_embedding"] = big_vision_vit.resample_posemb(
            old=restored_params["pos_embedding"],
            new=init_params["pos_embedding"],
        )

    return restored_params


def merge_params(loaded, inited, dont_load=(), match_dtype=False):
    """Merges checkpoint params into initialized params.

    For each initialized parameter:
    - if a matching checkpoint parameter exists,
    - and it is not excluded by `dont_load`,
    - and its shape matches,
      then the checkpoint value is used.

    Otherwise, the initialized value is kept.

    Extra checkpoint parameters that do not exist in the model are logged.
    Missing checkpoint parameters are also logged.

    Args:
        loaded: Parameters loaded from a checkpoint.
        inited: Parameters produced by model initialization.
        dont_load: Regex patterns for parameter names that should keep their
            initialized values.
        match_dtype: Whether to cast loaded values to the initialized dtype.

    Returns:
        A pytree with the structure of `inited`, using checkpoint values where
        appropriate and safe.
    """
    if inited is None:
        return loaded

    dont_load = u.check_and_compile_patterns(dont_load)

    def should_merge(name: str) -> bool:
        # We intentionally use regex search, not full-match, so patterns can
        # match path fragments anywhere in the parameter name.
        return not any(pattern.search(name) for pattern in dont_load)

    loaded_flat, _ = u.tree_flatten_with_names(loaded)
    inited_flat, _ = u.tree_flatten_with_names(inited)

    loaded_flat = {k: v for k, v in loaded_flat}
    inited_flat = {k: v for k, v in inited_flat}

    merged = {}

    for name, init_val in inited_flat.items():
        if name not in loaded_flat:
            logging.info("Parameter %s missing in checkpoint. Using init value.", name)
            merged[name] = init_val
            continue

        if not should_merge(name):
            logging.info("Ignoring checkpoint and using init value for %s", name)
            merged[name] = init_val
            continue

        loaded_val = loaded_flat[name]

        if hasattr(init_val, "shape") and hasattr(loaded_val, "shape"):
            if init_val.shape != loaded_val.shape:
                logging.info(
                    "Shape mismatch for %s: checkpoint %s vs init %s. Using init value.",
                    name,
                    loaded_val.shape,
                    init_val.shape,
                )
                merged[name] = init_val
                continue

        if match_dtype and hasattr(init_val, "dtype") and hasattr(loaded_val, "dtype"):
            loaded_val = loaded_val.astype(init_val.dtype)

        merged[name] = loaded_val

    def _pretty_print(title, names, indent="  "):
        if not names:
            return ""
        return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))

    not_in_loaded = inited_flat.keys() - loaded_flat.keys()
    not_in_inited = loaded_flat.keys() - inited_flat.keys()

    logging.info(_pretty_print("Parameters in model but not in checkpoint", not_in_loaded))
    logging.info(_pretty_print("Parameters in checkpoint but not in model", not_in_inited))

    # Only fail on unmatched keys that are not intentionally ignored on the
    # checkpoint side. Missing checkpoint keys are already handled by falling
    # back to init values above, so we do not raise for them here.
    not_in_inited = {k for k in not_in_inited if should_merge(k)}

    if not_in_inited:
        raise ValueError(
            _pretty_print("Params in checkpoint", loaded_flat.keys()) + "\n" +
            _pretty_print("Params in model (code)", inited_flat.keys()) + "\n" +
            _pretty_print(
                "Params in checkpoint but not in model (code) and not `dont_load`ed",
                not_in_inited,
                indent=" + ",
            )
        )

    return u.recover_tree(merged.keys(), merged.values())
