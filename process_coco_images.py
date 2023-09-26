"""
Process image folders and annotations from COCO format to filesystem-based format for training.
"""
import json
from pathlib import Path

from absl import app
from PIL import Image

flags = app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("images", None, "Path of images folder.", required=True)
flags.DEFINE_string("annotations", None, "Path of annotations file.", required=True)
flags.DEFINE_string("outdir", None, "Directory where images will be saved.", required=True)


def main(argv):
  images_path = Path(FLAGS.images)
  outdir_path = Path(FLAGS.outdir)

  with open(FLAGS.annotations) as fh:
    metadata = json.load(fh)

  images = {x["id"]: images_path / x["file_name"] for x in metadata["images"]}
  for annotation in metadata["annotations"]:
    l, t, w, h = annotation["bbox"]
    individual_id = annotation["name"]
    img = Image.open(images[annotation["image_id"]])
    img = img.crop((l, t, l + w, t + h))
    (outdir_path / individual_id).mkdir(parents=True, exist_ok=True)
    img.save(outdir_path / individual_id / (annotation["uuid"] + ".jpg"))


if __name__ == "__main__":
  try:
    app.run(main)
  except SystemExit:
    pass
