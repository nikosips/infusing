"""Main file for Scenic."""

from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections

from universal_embedding import app
from universal_embedding import classification_with_knn_eval_trainer
from universal_embedding import grain_datasets
from universal_embedding import models


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
    """Main function for Scenic."""

    data_rng, rng = jax.random.split(rng)

    dataset_dict = grain_datasets.get_training_dataset(
        config,
        data_rng=int(jax.device_get(data_rng)[0]),  # Needed for TPU backend compatibility.
    )

    model_cls = models.MODELS[config.model_class]

    classification_with_knn_eval_trainer.train(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset_dict=dataset_dict,
        workdir=workdir,
        writer=writer,
    )


if __name__ == "__main__":
    app.run(main=main)