import concurrent.futures
import itertools
import retry
from typing import Iterable

from absl import app
from absl import flags
from absl import logging
from google.cloud import aiplatform


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "action",
    "list_running",
    ["list_running", "list_failed", "cancel", "clear"],
    "What action to take on the jobs.",
)
flags.DEFINE_list(
    "location",
    ["us-central1"],
    "Google Cloud location(s) where the the experiment run.",
)
flags.DEFINE_list(
    "project",
    None,
    "Google Cloud project(s) where the the experiment run.",
    required=True,
)
flags.DEFINE_string(
    "experiment_title",
    "",
    "Experiment title i.e. job display name prefix.",
)


_CANCELLABLE_STATES = [
    aiplatform.gapic.JobState.JOB_STATE_PAUSED,
    aiplatform.gapic.JobState.JOB_STATE_PENDING,
    aiplatform.gapic.JobState.JOB_STATE_QUEUED,
    aiplatform.gapic.JobState.JOB_STATE_RUNNING,
    aiplatform.gapic.JobState.JOB_STATE_UPDATING,
]

_FAILED_STATES = [
    aiplatform.gapic.JobState.JOB_STATE_FAILED,
    aiplatform.gapic.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    aiplatform.gapic.JobState.JOB_STATE_UNSPECIFIED,
]


def _init_client(location: str) -> aiplatform.gapic.JobServiceClient:
  client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
  return aiplatform.gapic.JobServiceClient(client_options=client_options)


@retry.retry(tries=8, delay=1, backoff=2)
def _iter_custom_jobs(location: str, project: str) -> Iterable[aiplatform.gapic.CustomJob]:
  client = _init_client(location)
  yield from client.list_custom_jobs(parent=f"projects/{project}/locations/{location}")


@retry.retry(tries=8, delay=1, backoff=2)
def _cancel_custom_job(job: aiplatform.gapic.CustomJob) -> None:
  _, _, _, location, _, _ = job.name.split("/")
  client = _init_client(location)
  try:
    logging.info("[%s] Cancelling job %s.", location, job.display_name)
    client.cancel_custom_job(name=job.name)
    logging.info("[%s] Successfully cancelled job %s.", location, job.display_name)
  except Exception as exc:
    logging.error("[%s] Failed to cancel job %s. Error:\n%r", location, job.display_name, exc)


@retry.retry(tries=8, delay=1, backoff=2)
def _delete_custom_job(job: aiplatform.gapic.CustomJob) -> None:
  _, _, _, location, _, _ = job.name.split("/")
  client = _init_client(location)
  try:
    logging.info("[%s] Deleting job %s.", location, job.display_name)
    client.delete_custom_job(name=job.name)
    logging.info("[%s] Successfully deleted job %s.", location, job.display_name)
  except Exception as exc:
    logging.error("[%s] Failed to delete job %s. Error:\n%r", location, job.display_name, exc)
    raise exc


def main(_) -> None:
  action_count = 0
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for location, project in itertools.product(FLAGS.location, FLAGS.project):
      for job in _iter_custom_jobs(location, project):
        if job.display_name.startswith(FLAGS.experiment_title):
          if FLAGS.action == "list_running" and job.state in _CANCELLABLE_STATES:
            action_count += 1
            print(f"[{location}][{job.state.name}] {job.display_name}.")
          elif FLAGS.action == "list_failed" and job.state in _FAILED_STATES:
            action_count += 1
            print(f"[{location}][{job.state.name}] {job.display_name}.")
          elif FLAGS.action == "cancel" and job.state in _CANCELLABLE_STATES:
            action_count += 1
            executor.submit(_cancel_custom_job, job)
          elif FLAGS.action == "clear" and job.state not in _CANCELLABLE_STATES:
            action_count += 1
            executor.submit(_delete_custom_job, job)

  logging.info("Performed action %s affecting %d jobs.", FLAGS.action, action_count)


if __name__ == "__main__":
  app.run(main)
