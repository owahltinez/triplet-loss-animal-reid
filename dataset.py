"""
Helper methods to load datasets of images for triplet loss learning.
"""
import concurrent.futures
import functools
import math
import pathlib
import tempfile
from typing import Iterator, List, Tuple

from absl import logging
from google.cloud import storage

import numpy as np
import tensorflow as tf

IMAGE_SUFFIXES = ("jpg", "jpeg", "gif", "png")


def _iter_folder_images(path: pathlib.Path) -> Iterator[str]:
  yield from (p for p in path.iterdir() if p.is_file() and p.suffix[1:] in IMAGE_SUFFIXES)


def triplet_safe_image_dataset_from_directory(
    path: str,
    image_size: Tuple[int, int] = None,
    shuffle: bool = True,
    batch_size: int = 32,
    color_mode: str = "rgb",
    seed: float = None,
) -> tf.data.Dataset:
  assert color_mode in ("rgb", "rgba", "grayscale")

  # Instantiate the random number generator with known seed.
  rng = np.random.RandomState(seed=seed)

  # Inspect subdirectories and list all available images.
  subdirs = [d for d in pathlib.Path(path).iterdir() if d.is_dir()]
  class_names = [d.name for d in subdirs]
  class_labels = list(range(len(class_names)))
  paths_by_label = {i: list(_iter_folder_images(d)) for i, d in zip(class_labels, subdirs)}

  file_paths = list(sorted(sum(paths_by_label.values(), [])))
  logging.info(f"Found {len(file_paths)} files belonging to {len(class_names)} classes.")

  # Define inner functions that make use of provided arguments implicitly.
  def _read_image(file_path: str) -> tf.Tensor:
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3)
    if color_mode == "grayscale":
      img = tf.image.rgb_to_grayscale(img)
    if image_size is not None:
      img = tf.image.resize(img, image_size)
      img = tf.reshape(img, tuple([*image_size, -1]))
    return img

  def _iter_samples() -> Iterator[Tuple[pathlib.Path, int]]:
    for i, paths in paths_by_label.items():
      for p in paths:
        yield (p, i)

  def _sample_generator() -> Iterator[Tuple[tf.Tensor, int]]:
    samples = list(_iter_samples())
    for file_path_1, label in rng.permutation(samples) if shuffle else samples:
      possible_file_paths = [f for f in paths_by_label[label] if f != file_path_1]
      if possible_file_paths:
        file_path_2 = rng.choice(possible_file_paths)
        yield (_read_image(str(file_path_1)), label)
        yield (_read_image(str(file_path_2)), label)

  spec = (
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
      tf.TensorSpec(shape=(), dtype=tf.int32),
  )
  ds = tf.data.Dataset.from_generator(_sample_generator, output_signature=spec).batch(batch_size)

  # Set certain attributes similarly to tf.keras.utils.image_dataset_from_directory.
  setattr(ds, "file_paths", file_paths)
  setattr(ds, "class_names", class_names)
  return ds


def load_dataset(
    path: pathlib.Path,
    splits: List[float],
    batch_size: int = 32,
    seed: float = None,
    **kwargs,
) -> List[tf.data.Dataset]:
  assert abs(sum(splits) - 1) < 1e3, "Split fractions must add up to one"
  splits_cumsum = [sum(splits[:i]) for i in range(1, len(splits))]

  # Produce a list of all samples grouped by label.
  subdirs = [d.absolute() for d in pathlib.Path(path).iterdir() if d.is_dir()]

  rng = np.random.RandomState(seed=seed)
  split_indices = [math.ceil(frac * len(subdirs)) for frac in splits_cumsum]
  group_splits = list(np.split(rng.permutation(subdirs), split_indices))

  # Write the resulting splits into a temporary directory.
  tmpdir = pathlib.Path(tempfile.mkdtemp())
  for idx, group in enumerate(group_splits):
    (tmpdir / str(idx)).mkdir()
    for d in group:
      (tmpdir / str(idx) / d.name).symlink_to(d, target_is_directory=True)

  # Read the temporary directories into a tf dataset.
  ds_opts = dict(shuffle=True, color_mode="rgb", batch_size=batch_size, **kwargs)

  # Training and validation sets require distribution of samples that allow for triple mining.
  output = []
  for i in range(len(splits) - 1):
    img_dir = tmpdir / str(i)
    output.append(triplet_safe_image_dataset_from_directory(img_dir, **ds_opts))

  # Test set can use the normal sampling procedure.
  ds_test_func = tf.keras.utils.image_dataset_from_directory
  output.append(ds_test_func(tmpdir / str(len(splits) - 1), follow_links=True, **ds_opts))

  return output


def download_from_gcloud(
    bucket_name: str,
    output_dir: str,
    prefix: str = "",
    parallelism: int = 32,
) -> None:
  futures = []
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
    for blob in bucket.list_blobs(prefix=prefix):
      output_path = pathlib.Path(output_dir) / blob.name
      output_path.parent.mkdir(exist_ok=True, parents=True)
      future = executor.submit(blob.download_to_filename, output_path)
      futures.append(future)

  for future in concurrent.futures.as_completed(futures):
    if future.exception():
      raise future.exception()
