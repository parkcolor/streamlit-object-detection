import cv2
import tensorflow as tf
import os
import pathlib
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from glob import glob
# from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time

import streamlit as st
import pandas as pd

def ssd():
  PATH_TO_LABELS = 'C:\\Users\\5-15\\Documents\\GitHub\\streamlit-object-detection\\mscoco_label_map.pbtxt'
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
  
  def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
      fname=model_name, 
      origin=base_url + model_file,
      untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model

  # 함수 테스트
  model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
  detection_model =  load_model(model_name)

  # print(  detection_model.signatures['serving_default'].inputs   )
  # print(detection_model.signatures['serving_default'].output_dtypes)
  # print( detection_model.signatures['serving_default'].output_shapes )

  def run_inference_for_single_image(model, image):
    # 넘파이 어레이로 바꿔준다.
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
      output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])  
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
      
    return output_dict

  def show_inference(model, image):
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.

      # image_np = np.array(Image.open(image_path))
      # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

      image_np = np.asarray(image)

      # Actual detection.
      output_dict = run_inference_for_single_image(model, image)
      # Visualization of the results of a detection.

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.array(output_dict['detection_boxes']),
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed',None),
          use_normalized_coordinates=True,
          line_thickness=8)
      
      start_time = time.time()
      st.image(image_np,channels="BGR")
      end_time = time.time()
      print(end_time - start_time)

  upload_file = st.file_uploader('이미지를 업로드하세요',type=['png','jpg','jpeg'])
  if upload_file is not None:
    file_byte = np.asarray(bytearray(upload_file.read()), dtype= np.uint8)
    image = cv2.imdecode(file_byte, 1)
    show_inference(detection_model, image)




