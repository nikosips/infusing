from typing import Any, Callable, Iterable

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from universal_embedding.mlp import Mlp


Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
JAX_PRECISION = "default"


class EmbeddingHead(nn.Module):
    output_dim: int = -1
    dataset_meta_data: Any = None
    config: ml_collections.ConfigDict = None

    @nn.compact
    def __call__(
        self,
        outputs: Any,
        domain: int,
        *,
        train: bool,
        debug: bool = False,
        init: bool = False,
        return_feats: bool = False,
        domain_agnostic: bool = False,
    ):
        """Builds projected embeddings and optional classifier logits.

        Args:
            outputs: Model output dictionary containing an "embeddings" entry.
            domain: Dataset/domain index for the current batch.
            train: Whether the model is in training mode.
            debug: Unused, kept for interface compatibility.
            init: Unused, kept for interface compatibility.
            return_feats: If True, skip classifier head computation.
            domain_agnostic: Unused, kept for interface compatibility.

        Returns:
            Updated outputs dictionary with encoded embeddings and optional logits.
        """
        del train, debug, init, domain_agnostic

        outputs.setdefault("classifier", {})

        backbone_out = outputs["embeddings"]["backbone_out_embedd"]
        if self.config.model.stopgrad_backbone_out_to_encoded:
            backbone_out = jax.lax.stop_gradient(backbone_out)

        encoded_embedd = Mlp(
            num_layers=len(self.config.model.encoder_mlp_dim),
            hidden_size_list=self.config.model.encoder_mlp_dim,
            name="encoder_projection_domain_0",
            skip_connect=self.config.model.use_skip_on_mlp,
        )(backbone_out)

        # L2-normalize encoded embeddings safely.
        encoded_norm = jnp.linalg.norm(encoded_embedd, ord=2, axis=1, keepdims=True)
        encoded_norm = jnp.maximum(encoded_norm, 1e-12)
        encoded_embedd = encoded_embedd / encoded_norm

        outputs["embeddings"]["encoded_embedd"] = encoded_embedd

        domain_key = str(domain)

        if domain_key != "-1": #special case for domain-agnostic knn where domain is set to -1
            classif_losses_on = self.config.loss.classif_losses_on[domain_key]
        else:
            classif_losses_on = ""

        if return_feats or classif_losses_on == "":
            return outputs

        dataset_names = [name.strip() for name in self.dataset_meta_data["dataset_name"].split(",")]
        current_dataset = dataset_names[domain]

        if self.config.classifier == "separate":
            classifier_domain = domain
            classifier_num_classes = self.dataset_meta_data["classes_per_dataset"][current_dataset]
        elif self.config.classifier == "joint":
            classifier_domain = 0
            classifier_num_classes = self.dataset_meta_data["num_classes"]
        else:
            raise ValueError(f"Classifier type {self.config.classifier} not supported.")

        loss_names = [name.strip() for name in classif_losses_on.split(",") if name.strip()]
        stopgrad_losses = {
            name.strip()
            for name in self.config.loss.stopgrad_on_classifier_on[domain_key].split(",")
            if name.strip()
        }

        for classif_loss_on in loss_names:
            classif_input = outputs["embeddings"][f"{classif_loss_on}_embedd"]

            if classif_loss_on in stopgrad_losses:
                classif_input = jax.lax.stop_gradient(classif_input)

            logits = nn.Dense(
                classifier_num_classes,
                use_bias=False,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"{classif_loss_on}_output_projection_{classifier_domain}",
            )(classif_input)

            kernel = self.variables["params"][
                f"{classif_loss_on}_output_projection_{classifier_domain}"
            ]["kernel"]
            weights_norms = jnp.linalg.norm(kernel, axis=0)
            weights_norms = jnp.maximum(weights_norms, 1e-12)
            logits = logits / weights_norms

            if self.config.loss.trainable_scale:
                scalar = self.param(
                    f"{classif_loss_on}_logit_scale_{classifier_domain}",
                    nn.initializers.ones,
                    (1,),
                )
                logits = logits * (self.config.loss.scale * scalar)

            outputs["classifier"][f"{classif_loss_on}_logits"] = logits

        return outputs