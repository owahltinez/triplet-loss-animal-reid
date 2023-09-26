import itertools
import random
import time

from tqdm.auto import tqdm
from absl import app
from absl import flags
from absl import logging
from xmanager import xm
from xmanager import xm_local


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset",
    None,
    "Datasets to run experiments on.",
    required=True,
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Root directory where results will be stored.",
    required=True,
)
flags.DEFINE_list(
    "location",
    ["us-central1"],
    "Google Cloud locations to run the experiment.",
)
flags.DEFINE_string(
    "project",
    None,
    "Google Cloud project to run the experiment.",
    required=True,
)
flags.DEFINE_string(
    "experiment_title",
    "experiment",
    "Title to use for the experiment i.e. job display name prefix.",
)
flags.DEFINE_integer(
    "max_trials",
    100,
    "Maximum number of hyper-parameter combinations to try.",
)

# Fixed experiment parameters.
flags.DEFINE_integer("train_epochs", 50, "Total training epochs.")
flags.DEFINE_integer("seed", 0, "Seed used for various random number generators.")

# Hyper-parameters for transfer learning fine-tuning.
flags.DEFINE_list(
    "batch_size",
    [16, 32, 64, 128],
    "Number of samples per batch.",
)
flags.DEFINE_list(
    "learning_rate",
    [0.001],
    "Rate of learning during training.",
)
flags.DEFINE_list(
    "dropout",
    [0.0, 0.1, 0.2, 0.3],
    "Dropout regularization factor applied during training.",
)
flags.DEFINE_list(
    "augmentation_count",
    [2, 4, 6, 8],
    "Number of augmentations to apply per image.",
)
flags.DEFINE_list(
    "augmentation_factor",
    [0.0, 0.1, 0.2, 0.3],
    "Factor of image augmentation.",
)
flags.DEFINE_list(
    "loss_margin",
    [0.10, 0.25, 0.50, 0.75],
    "Margin used for semi-hard triplet loss.",
)
flags.DEFINE_list(
    "embedding_size",
    [64, 128, 256, 512],
    "Output embedding dimensions.",
)
flags.DEFINE_list(
    "retrain_layer_count",
    [64, 128, 200, 512],
    "Number of layers to retrain from base model.",
)
flags.DEFINE_list(
    "vote_count",
    [5, 10],
    "Number of votes per embedding in closed eval mode.",
)


@xm.run_in_asyncio_loop
async def main(_) -> None:
  experiment_opts = dict(experiment_title=FLAGS.experiment_title)
  async with xm_local.create_experiment(**experiment_opts) as experiment:
    [executable] = experiment.package(
        [
            xm.python_container(
                entrypoint=xm.ModuleName("experiment"),
                base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-11.py310",
                executor_spec=xm_local.Vertex.Spec(),
            )
        ]
    )

    hyper_params = dict(
        embedding_size=FLAGS.embedding_size,
        loss_margin=FLAGS.loss_margin,
        dropout=FLAGS.dropout,
        batch_size=FLAGS.batch_size,
        augmentation_factor=FLAGS.augmentation_factor,
        augmentation_count=FLAGS.augmentation_count,
        retrain_layer_count=FLAGS.retrain_layer_count,
    )

    # Shuffle the possible locations.
    locations = list(FLAGS.location)
    random.shuffle(locations)

    # Pick only a random subset of hyper-parameter combinations.
    param_sets = list(itertools.product(*hyper_params.values()))
    random.shuffle(param_sets)
    param_values = param_sets[: FLAGS.max_trials]

    logging.info("Running experiments on dataset: %r.", FLAGS.dataset)
    for idx, values in enumerate(tqdm(param_values)):
      params = dict(zip(hyper_params.keys(), values))

      # Use experiment + timestamp + random number for the file name.
      fname = f"{FLAGS.experiment_title}"
      fname += f"-{int(time.time())}"
      fname += f"-{int(random.random() * 1_000):04d}.zip"

      # Round-robin selection for the experiment location.
      location = locations[idx % len(locations)]
      client = xm_local.vertex.Client(project=FLAGS.project, location=location)
      xm_local.vertex.set_default_client(client)

      # Add the job to the experiment and await until it's launched.
      await experiment.add(
          xm.Job(
              name=str(params),
              executable=executable,
              executor=xm_local.Vertex(
                  requirements=xm.JobRequirements(
                      location=location,
                      resources={
                          # Matches n1-standard-32.
                          xm.ResourceType.CPU: 32 * xm.vCPU,
                          xm.ResourceType.RAM: 120 * xm.GiB,
                      },
                  ),
              ),
              args=dict(
                  dataset=FLAGS.dataset,
                  output=f"{FLAGS.output_dir}/{fname}",
                  train_epochs=FLAGS.train_epochs,
                  seed=FLAGS.seed,
                  **params,
              ),
          )
      )


if __name__ == "__main__":
  app.run(main)
