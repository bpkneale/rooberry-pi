
from aws.sqs import TelemetryQueue, send_message
from workers.base import Worker
import time
import logging
import json
import datetime

_log = logging.getLogger(__name__)

is_windows = False
try:
  import pijuice
except ImportError:
  is_windows = True

StatusFns = [
  'GetBatteryCurrent',
  'GetBatteryTemperature',
  'GetBatteryVoltage',
  'GetButtonEvents',
  'GetChargeLevel',
  'GetFaultStatus',
  # 'GetIoAnalogInput',
  'GetIoCurrent',
  # 'GetIoDigitalInput',
  # 'GetIoDigitalOutput',
  # 'GetIoPWM',
  'GetIoVoltage',
  # 'GetLedBlink',
  # 'GetLedState',
  'GetStatus'
]

class PiJuiceWorker(Worker):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.pj = None
    if not is_windows:
      self.pj = pijuice.PiJuice(1, 0x14)

  def fetch_pijuice_stuff(self):
    vals = {}
    for fn_name in StatusFns:
      fn = getattr(self.pj.status, fn_name)
      vals[fn_name] = fn()
    return vals

  def run(self) -> None:
    while self._run_event.is_set():
      _log.debug("Fetching pijuice info")
      if self.pj:
        status = self.fetch_pijuice_stuff()
        _log.info(status)
        send_message(TelemetryQueue, json.dumps({
          "roo_camera": "roopi1/telemetry",
          "generated_at": datetime.datetime.utcnow().isoformat(),
          "status": status
        }))
      else:
        _log.debug("Running under windows :(")
      time.sleep(60.0)


def main():
  worker = PiJuiceWorker()
  print(worker.fetch_pijuice_stuff())


if __name__ == '__main__':
  main()
