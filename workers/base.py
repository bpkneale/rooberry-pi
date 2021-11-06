from threading import Event, Thread
from logging import Logger, DEBUG

class Worker(Thread):
  def __init__(self, *args, **kwargs) -> None:
    self._run_event = Event()
    self._run_event.set()
    super().__init__(*args, **kwargs)

  def stop(self):
    self._run_event.clear()
    self.join(5.0)
