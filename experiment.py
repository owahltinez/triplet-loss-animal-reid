"""
Trains a model using the triplet learning architecture.
"""
import json
import time
import pathlib
import shutil
import tempfile

import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import dataset
import mean_average_precision

from absl import app
from absl import logging

flags = app.flags
FLAGS = flags.FLAGS

# I/O parameters.
flags.DEFINE_string(
    "dataset",
    None,
    "Path to image dataset, either locally available or in GCS.",
    required=True,
)
flags.DEFINE_string(
    "output",
    None,
    "Zip file where experiment results will be stored.",
    required=True,
)

# Hyper-parameters for transfer learning fine-tuning.
flags.DEFINE_integer("batch_size", 32, "Number of samples per batch.")
flags.DEFINE_float("learning_rate", 0.001, "Rate of learning during training.")
flags.DEFINE_float("dropout", 0.1, "Dropout regularization factor applied during training.")
flags.DEFINE_integer("augmentation_count", 4, "Number of augmentations to apply per image.")
flags.DEFINE_float("augmentation_factor", 0.1, "Factor of image augmentation.")
flags.DEFINE_float("loss_margin", 0.5, "Margin used for semi-hard triplet loss.")
flags.DEFINE_integer("embedding_size", 128, "Output embedding dimensions.")
flags.DEFINE_integer("retrain_layer_count", 128, "Number of layers to retrain from base model.")

# Experiment-related parameters.
flags.DEFINE_integer("train_epochs", 50, "Total training epochs.")
flags.DEFINE_bool("save_model", False, "Whether to save the trained model.")
flags.DEFINE_integer("vote_count", 5, "Number of votes per embedding in closed eval mode.")
flags.DEFINE_integer("verbose", 1, "Verbosity level used for keras model fit.")
flags.DEFINE_integer("seed", 0, "Seed used for various random number generators.")


INPUT_SHAPE = (224, 224, 3)


def _preprocessing_layer(x: tf.keras.layers.Input):
  x = tf.cast(x, tf.float32)
  x = tf.keras.applications.densenet.preprocess_input(x)
  return x


def _augmentation_layer(
    x: tf.keras.layers.Input,
    augmentation_count: int,
    augmentation_factor: float,
):
  n = augmentation_count
  k = augmentation_factor

  # Early exit: no augmentations necessary.
  if n == 0 or k < 1e-6:
    return x

  t_opts = dict(seed=FLAGS.seed)
  transformations = [
      tf.keras.layers.Dropout(k, **t_opts),
      tf.keras.layers.GaussianNoise(k, **t_opts),
      tf.keras.layers.RandomFlip("horizontal", **t_opts),
      tf.keras.layers.RandomTranslation(k, k, fill_mode="constant", **t_opts),
      tf.keras.layers.RandomRotation(k, fill_mode="constant", **t_opts),
      tf.keras.layers.RandomZoom(k, k, fill_mode="constant", **t_opts),
      keras_cv.layers.RandomHue(k, [0, 255], **t_opts),
      keras_cv.layers.RandomSaturation(k, **t_opts),
  ]

  for t in np.random.choice(transformations, size=n, replace=False):
    x = t(x)

  return x


def _inference_layer(
    x: tf.keras.layers.Input,
    dropout: float,
    embedding_size: int,
):
  # Used to initialize the weights of untrained layers.
  w = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  if dropout > 0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Dense(embedding_size, activation=None, bias_initializer=w)(x)
  x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedding")(x)

  return x


def _transfer_layer(
    x: tf.keras.layers.Input,
    retrain_layer_count: int,
):
  model = tf.keras.applications.DenseNet121(
      include_top=False,
      weights="imagenet",
      input_shape=INPUT_SHAPE,
  )

  # Enable training only for the selected layers of the base model.
  for layer in model.layers[:-retrain_layer_count]:
    layer.trainable = False

  # Run the input through the model.
  x = model(x)

  return x


def _build_model(
    dropout: float,
    augmentation_count: int,
    augmentation_factor: float,
    embedding_size: int,
    retrain_layer_count: int,
    loss_margin: float,
    learning_rate: float,
) -> tf.keras.Model:
  # Initialize the input parameters.
  x = inputs = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.uint8)

  # Run the inputs through the model's layers.
  x = _preprocessing_layer(x)
  x = _augmentation_layer(x, augmentation_count, augmentation_factor)
  x = _transfer_layer(x, retrain_layer_count)
  x = _inference_layer(x, dropout, embedding_size)

  # Encapsulate the I/O into a model type.
  model = tf.keras.Model(inputs=[inputs], outputs=x)

  # Compile the model and return it.
  model.compile(
      loss=tfa.losses.TripletSemiHardLoss(soft=False, margin=loss_margin),
      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
  )

  return model


def main(_):
  # Validate required parameters.
  if not str(FLAGS.output).endswith(".zip"):
    raise ValueError(f'Parameter "output" must be a path to a zip file: {FLAGS.output}.')

  # Create a folder to save progress and results.
  output_root = pathlib.Path(tempfile.mkdtemp())
  experiment_path = output_root / "experiment"
  experiment_path.mkdir(parents=True, exist_ok=True)
  logging.info(f"Using {output_root} to store temporary experiment results.")

  # Download the dataset if it doesn't exist locally.
  dataset_name = FLAGS.dataset.split("/")[-1]
  if pathlib.Path(FLAGS.dataset).exists():
    dataset_path = pathlib.Path(FLAGS.dataset)
    logging.info("Dataset %s found in local disk.", dataset_name)
  elif FLAGS.dataset.startswith("gs://"):
    dataset_path = output_root / dataset_name
    logging.info("Downloading dataset %s into %s.", dataset_name, str(dataset_path))
    bucket_name = FLAGS.dataset[len("gs://") :].split("/", 1)[0]
    dataset.download_from_gcloud(
        bucket_name=bucket_name,
        output_dir=dataset_path.parent,
        prefix=FLAGS.dataset[len("gs://") + len(bucket_name) + 1 :] + "/",
    )
  else:
    raise ValueError(f"Unknown dataset location: {FLAGS.dataset}")

  # Retrieve parameters from flags.
  parameters = dict(
      dropout=FLAGS.dropout,
      augmentation_count=FLAGS.augmentation_count,
      augmentation_factor=FLAGS.augmentation_factor,
      learning_rate=FLAGS.learning_rate,
      embedding_size=FLAGS.embedding_size,
      retrain_layer_count=FLAGS.retrain_layer_count,
      loss_margin=FLAGS.loss_margin,
  )

  # Start experiment.
  time_start = time.perf_counter()
  logging.info(f"Starting experiment for {dataset_name}.")
  results = dict(dataset=dataset_name)

  # Build and compile model using provided parameters.
  model = _build_model(**parameters)

  # Add non-model parameters to the parameter dictionary.
  parameters["dataset"] = dataset_name
  parameters["batch_size"] = FLAGS.batch_size
  parameters["train_epochs"] = FLAGS.train_epochs

  # Load dataset from filesystem.
  if (dataset_path / "train").exists() and (dataset_path / "test").exists():
    # Datasets might already be split between train and test.
    ds_opts = dict(
        image_size=list(INPUT_SHAPE)[:-1],
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
    )
    train_data = dataset.triplet_safe_image_dataset_from_directory(
        dataset_path / "train",
        **ds_opts,
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "test",
        **ds_opts,
    )
    validation_data = (
        tf.keras.utils.image_dataset_from_directory(
            dataset_path / "validation",
            **ds_opts,
        )
        # If there's no specific validation subset, re-use test data.
        if (dataset_path / "validation").exists()
        else test_data
    )

  else:
    # Otherwise we will need to create our own test subset.
    train_data, validation_data, test_data = dataset.load_dataset(
        dataset_path,
        [0.7, 0.1, 0.2],
        image_size=list(INPUT_SHAPE)[:-1],
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
    )

  # Determine if it's closed or open eval based on whether all labels from the
  # training set are also present in the test set.
  if all(label in test_data.class_names for label in train_data.class_names):
    model_eval = "closed"
  else:
    model_eval = "open"
  parameters["model_eval"] = model_eval
  logging.info("Using model evaluation: %s set.", model_eval)

  # Write the subset labels to disk.
  datasets = train_data, validation_data, test_data
  for name, subset in zip(["train", "validation", "test"], datasets):
    if subset is not None:
      subset_labels = subset.class_names
      logging.info("Using %d classes for subset %s.", len(subset_labels), name)
      with open(experiment_path / f"{name}_subset.txt", "w") as f:
        f.write("\n".join({label for label in subset_labels}))

  # Write out the model's summary to disk.
  logging.info("Writing model summary.")
  with open(experiment_path / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
    for layer in model.layers:
      if hasattr(layer, "summary"):
        layer.summary(print_fn=lambda x: f.write(x + "\n"))

  # Fit the model for the entire number of epochs unless early stopping.
  callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
      tf.keras.callbacks.TensorBoard(log_dir=experiment_path / f"log"),
  ]

  # Compute the mAP each epoch using open set eval mode only (faster).
  cb_subset = validation_data if validation_data else test_data
  mAP_callback = mean_average_precision.MeanAveragePrecisionCallback
  callbacks.append(mAP_callback(top_k=[1, 5], dataset=cb_subset))

  # Fit the model.
  history = model.fit(
      x=train_data.cache().prefetch(tf.data.AUTOTUNE),
      validation_data=validation_data.cache().prefetch(tf.data.AUTOTUNE),
      epochs=FLAGS.train_epochs,
      verbose=FLAGS.verbose,
      callbacks=callbacks,
  )

  # Write out the used parameters.
  with open(experiment_path / "parameters.json", "w") as f:
    json.dump(parameters, f)

  # Save the total number of epochs run.
  results["epochs"] = len(history.history["loss"])

  # Record into results the last metric values from the history.
  for key in history.history.keys():
    results[key] = history.history[key][-1]

  # Compute elapsed time for training loop.
  results["time_train"] = time.perf_counter() - time_start

  # Compute mean average precision metrics and save them to disk.
  map_top_k = (1, 5)
  logging.info("Computing mAP@%r using %s set eval.", map_top_k, model_eval)
  if model_eval == "open":
    results.update(
        mean_average_precision.evaluate_model_open_set(
            model,
            test_data,
            top_k=map_top_k,
        )
    )
  elif model_eval == "closed":
    results.update(
        mean_average_precision.evaluate_model_closed_set(
            model,
            train_data,
            test_data,
            top_k=map_top_k,
            vote_count=FLAGS.vote_count,
        )
    )

  # Compute elapsed time for entire trial, including evaluation.
  results["time_trial"] = time.perf_counter() - time_start
  results["time_eval"] = results["time_trial"] - results["time_train"]

  # Log results and save them to disk.
  logging.info("Experiment %s results: %r", dataset_name, results)
  with open(experiment_path / "results.json", "w") as f:
    json.dump(results, f)

  # Save the model into a file for later analysis.
  if FLAGS.save_model:
    logging.info("Saving model into output folder.")
    model.save(experiment_path / f"model")

  # Zip results and copy them to final output destination.
  logging.info("Writing output to %s.", FLAGS.output)
  output_tmp_zip = f"{experiment_path}.zip"
  shutil.make_archive(experiment_path, "zip", experiment_path)
  with (
      open(output_tmp_zip, "rb") as f_in,
      tf.io.gfile.GFile(FLAGS.output, "wb") as f_out,
  ):
    shutil.copyfileobj(f_in, f_out)

  # Remove temporary files.
  logging.info("Removing temporary files at %s.", output_root)
  shutil.rmtree(output_root)


if __name__ == "__main__":
  app.run(main)
