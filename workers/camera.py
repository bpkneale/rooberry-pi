import sys
from typing import Optional
from .base import Worker
import json
import tflite_runtime.interpreter as tflite
import time
import numpy as np
from PIL import Image, ImageDraw
import boto3
from botocore.exceptions import ClientError
import os
import exif
import uuid
import requests
import logging
import math
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)

is_windows = False
camera = None
try:
  from picamera import PiCamera
except ImportError:
  is_windows = True


default_bucket = 'roo-pie'
# Upload the file
s3_client = boto3.client('s3',
  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)
ApiBase = "https://3tppyzkh4g.execute-api.ap-southeast-2.amazonaws.com"

class CameraWorker(Worker):
  def __init__(self, *args, **kwargs) -> None:
    self._override_image = kwargs.pop('override_image', None)
    super().__init__(*args, **kwargs)
    self._last_periodic = datetime.min

  def upload_file(self, file_name, inference_results, kind: str):
    object_name = f"{uuid.uuid4()}-{os.path.basename(file_name)}"
    try:
        s3_client.upload_file(file_name, default_bucket, object_name)
    except ClientError as e:
        _log.info(f"Error uploading file: {e}")
    else:
      resp = requests.post(f"{ApiBase}/sighting", json={
        "rooCamera": f"roopi1/{kind}",
        "captureTimeMs": str(time.time() * 1000),
        "s3Key": object_name,
        "inferenceResults": json.dumps(inference_results)
      })
      _log.info(resp.json())


  def load_labels(self, filename):
    with open(filename, 'r') as f:
      return [line.strip() for line in f.readlines()]


  def get_image(self, resize_to):
    img = None
    filename = self._override_image
    if not is_windows:
      with PiCamera(resolution="1080p") as camera:
        camera.awb_mode = "incandescent"
        _log.info("Sleeping for camera to get balanced")
        time.sleep(2.0)
        # camera.saturation = -50
        filename = f"/tmp/capture-{round(time.time() * 1000)}.jpg"
        camera.capture(filename)

    img = Image.open(filename)
    
    sub_images = []
    bounding_boxes = []
    source_width, source_height = img.size
    target_width, target_height = resize_to
    horizontal_zones = source_width / target_width
    vertical_zones = source_height / target_height
    horizontal_images = math.ceil(horizontal_zones) + 1
    vertical_images = math.ceil(vertical_zones) + 1

    _log.info(f"HZ: {horizontal_zones}, HI: {horizontal_images}, VZ: {vertical_zones}, VI: {vertical_images}")

    if horizontal_zones > 1 and vertical_zones > 1:
      for vi in range(vertical_images):
        for hi in range(horizontal_images):
          left = math.floor(hi * ((horizontal_zones / horizontal_images) * target_width))
          upper = math.floor(vi * ((vertical_zones / vertical_images) * target_height))
          right = left + target_width
          lower = upper + target_height
          crop_area = (left, upper, right, lower)
          _log.info(f"Creating crop area {crop_area}")
          sub_images.append(img.crop(crop_area))
          bounding_boxes.append(crop_area)
    else:
      sub_images.append(img.resize(resize_to))
      bounding_boxes.append((0, 0, source_width, source_height))

    return sub_images, bounding_boxes, filename


  def upload_result(self, image_inference_results, filepath: str, kind: str, detect_boxes):
    _log.info(f"Uploading inference success for path {filepath}")
    splot = filepath.split(".")
    new_name = splot[0] + '-with-boxes.' + splot[1]
    out_file = filepath

    if detect_boxes:
      with Image.open(filepath) as im:
        for box in detect_boxes:
          draw = ImageDraw.Draw(im)
          draw.rectangle(box, outline=128)
          im.save(new_name)
      os.remove(filepath)
      out_file = new_name

    self.upload_file(out_file, image_inference_results, kind)


  def is_maybe_a_kangaroo(self, inference_result, label: str):
    return inference_result > 0.25 and label.count('kangaroo') > 0
    
  def run(self) -> None:
    interpreter = tflite.Interpreter(model_path='mobilenet_v1_1.0_224.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    _log.info(f"Floating model: {floating_model}")

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    labels = self.load_labels("labels.txt")

    while self._run_event.is_set():
      sub_images, bounding_boxes, filepath = self.get_image((width, height))
      kangaroo_sighting = False
      detect_boxes = []
      image_inference_results = []

      for img, box in zip(sub_images, bounding_boxes):
        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
          input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        _log.info(results)
        _log.info(output_data)
        top_k = results.argsort()[-5:][::-1]
        inference_results = []
        for i in top_k:
          normalized = results[i] if floating_model else results[i] / 255.0
          inference_result = '{:08.6f}: {}'.format(float(normalized), labels[i])
          this_one = self.is_maybe_a_kangaroo(normalized, labels[i])

          if this_one:
            detect_boxes.append(box)

          kangaroo_sighting = kangaroo_sighting or this_one
          inference_results.append(inference_result)

        image_inference_results.append({
          "bounding_box": [int(b) for b in box],
          "inference_results": inference_results
        })
        
      _log.info(inference_results)

      if is_windows:
        self.upload_result(image_inference_results, filepath, "test", detect_boxes)
        break

      if kangaroo_sighting:
        self.upload_result(image_inference_results, filepath, "sighting", detect_boxes)

      _log.info('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

      delta: timedelta = datetime.now() - self._last_periodic
      if delta.total_seconds() > 60 * 15:
        _log.info(f'retaining capture {filepath}')
        self.upload_result(image_inference_results, filepath, "periodic", detect_boxes)
        self._last_periodic = datetime.now()
      else:
        _log.info('removing capture')
        os.remove(filepath)


def main():
  import sys
  logging.basicConfig(level=logging.DEBUG)
  worker = CameraWorker(override_image=sys.argv[1])
  worker.run()

if __name__ == '__main__':
  main()
