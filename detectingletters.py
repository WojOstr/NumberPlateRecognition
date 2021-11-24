import numpy as np
import easyocr
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from rotateimgscript import rotate
from deskew import determine_skew
import cv2
import string

ALLOWED_LIST = string.ascii_uppercase+string.digits

def load_model():
    gpus = tf.config.list_physical_devices('GPU');

    if gpus:
        try: 

            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2060)])
        except RunTimeError as e:
            print(e)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(r"Tensorflow\workspace\models\my_mobilenet_model_v2\pipeline.config")
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(r"Tensorflow\workspace\models\my_mobilenet_model_v2\ckpt-11").expect_partial()

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

    # detection_classes should be ints.
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
        min_score_thresh=.7,
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
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    
    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]
    region = []
    text = []
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region.append(image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])])
        reader = easyocr.Reader(['pl'])
        ocr_result = reader.readtext(region[idx], decoder = 'beamsearch', beamWidth = 10, allowlist=ALLOWED_LIST, min_size = 10, width_ths = 1.5)
        print(ocr_result)
        text.append(filter_text(region[idx], ocr_result))
        
    return text, region
"""
def ocr_it(image, detections, detection_threshold, region_threshold):
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]

        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['pl'])
    
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result)
        if len(text) > 1:
            strlist = ''.join([str(elem) for elem in text])
            if strlist.isalnum():
                text = list(strlist)
            else:
                grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                angle = determine_skew(grayscale)
                if angle:
                    rotated_image = rotate(image, angle, (0,0,0))
                    roi = box * calculate_new_roi(roi, angle, image)
                    region = rotated_image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
                    ocr_result = reader.readtext(region)
                    text = filter_text(region, ocr_result)

        elif text[0].isalnum() == False:
            grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(grayscale)
            if angle:
                rotated_image = rotate(image, angle, (0,0,0))
                roi = box * calculate_new_roi(roi, angle, image)

                region = rotated_image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
                ocr_result = reader.readtext(region)
                text = filter_text(region, ocr_result)

        ## SPRAWDZIĆ CZY MA BIAŁE ZNAKI
        ## ISALPHANUM
        ## SPŁASZCZYĆ W JEDNOŚĆ

        return text, region


def calculate_new_roi(roi, angle, image):
    width = image.shape[1]
    height = image.shape[0]
    angle = math.radians(angle)
    y1,x1,y2,x2 = roi[0],roi[1],roi[2],roi[3]
    print(width, height)

    x1_new = (x1-width/2) * math.cos(angle) - (y1 - height/2)* math.sin(angle) + width
    y1_new = (x1-width/2) * math.sin(angle) + (y1 - height/2) * math.cos(angle) + height

    x2_new = (x2-width/2) * math.cos(angle) - (y2 - height/2)* math.sin(angle) + width
    y2_new = (x2-width/2) * math.sin(angle) + (y2 - height/2) * math.cos(angle) + height

    coordinates = [math.fabs(y1_new), math.fabs(x1_new), math.fabs(y2_new), math.fabs(x2_new)]

    print(coordinates)
    return coordinates
    """