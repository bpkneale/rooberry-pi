import time
import logging
from workers import camera, pijuiceworker, base

def setup_logging():
  logging.basicConfig(level=logging.DEBUG)
  return logging.getLogger(__name__)

def main():
  logger = setup_logging()

  worker_threads: list[base.Worker] = [
    camera.CameraWorker(),
    pijuiceworker.PiJuiceWorker()
  ]

  for thread in worker_threads:
    thread.start()

  loop = True 
  while loop:
    for thread in worker_threads:
      if not thread.is_alive():
        logger.error(f"Thread ${thread} is no longer alive")
        loop = False
    time.sleep(1.0)

  for thread in worker_threads:
    if thread.is_alive():
      logger.info(f"stopping thread {thread}")
      thread.stop()

  logger.info("fin")

if __name__ == '__main__':
  main()
