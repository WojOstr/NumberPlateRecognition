import numpy as np
import easyocr
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from deskew import determine_skew
import cv2
import string
import math

ALLOWED_LIST = string.ascii_uppercase+string.digits

def load_model():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try: 

            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2060)])
        except RunTimeError as e:
            print(e)

    configs = config_util.get_configs_from_pipeline_file(r"Tensorflow\workspace\models\my_mobilenet_model_v2\export\pipeline.config")
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(r"Tensorflow\workspace\models\my_mobilenet_model_v2\export\checkpoint\ckpt-0").expect_partial()
    return detection_model


@tf.function(experimental_relax_shapes=True)
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detecting(sourceImage, detection_model):
    category_index = label_map_util.create_category_index_from_labelmap(r"Tensorflow\workspace\annotations\label_map.pbtxt")

    image_np = np.array(sourceImage)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.6,
        agnostic_mode=False)
    return image_np_with_detections, detections


def filter_text(region, ocr_result):
    region_threshold = 0.2
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def ocr_it(image, detections, detection_threshold):
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    region = []
    text = []

    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        temp_region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['pl'])

        ocr_result = reader.readtext(temp_region, decoder = 'beamsearch', beamWidth = 10, allowlist=ALLOWED_LIST, min_size = 10, width_ths = 1.5)
        
        grayscale = cv2.cvtColor(temp_region, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        if angle and (angle > 10 or angle < -10):
            center = (width/2, height/2)
            rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
            roi = calculate_new_roi(roi, angle, image, rotated_image)
            temp_region = rotated_image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            ocr_result = reader.readtext(temp_region, decoder = 'beamsearch', beamWidth = 10, allowlist=ALLOWED_LIST, min_size = 10, width_ths = 1.5)

        region.append(temp_region)
        text.append(filter_text(region[idx], ocr_result))
    
    return text, region


def calculate_new_roi(roi, angle, image, rotated_image):
    width = image.shape[1]
    height = image.shape[0]
    rotated_width = rotated_image.shape[1]
    rotated_height = rotated_image.shape[0]

    angle = math.radians(angle) * (-1)
    y1,x1,y2,x2 = roi[0],roi[1],roi[2],roi[3]

    x1_new = (x1-width/2) * math.cos(angle) - (y1 - height/2) * math.sin(angle) + rotated_width / 2
    y1_new = (x1-width/2) * math.sin(angle) + (y1 - height/2) * math.cos(angle) + rotated_height /  2

    x2_new = (x2-width/2) * math.cos(angle) - (y2 - height/2) * math.sin(angle) + rotated_width / 2
    y2_new = (x2-width/2) * math.sin(angle) + (y2 - height/2) * math.cos(angle) + rotated_height / 2

    coordinates = [math.fabs(y1_new), math.fabs(x1_new), math.fabs(y2_new), math.fabs(x2_new)]

    return coordinates