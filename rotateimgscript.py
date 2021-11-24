import cv2
import numpy as np
import math


def rotate(image, angle, background):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def calculate_new_roi(roi, angle, image):
    width = image.shape[1]
    height = image.shape[0]
    angle = math.radians(angle)
    y1,x1,y2,x2 = roi[0],roi[1],roi[2],roi[3]
    
    x1_new = (x1-width/2) * math.cos(angle) - (y1 - height/2)* math.sin(angle) + width
    y1_new = (x1-width/2) * math.sin(angle) + (y1 - height/2) * math.cos(angle) + height

    x2_new = (x2-width/2) * math.cos(angle) - (y2 - height/2)* math.sin(angle) + width
    y2_new = (x2-width/2) * math.sin(angle) + (y2 - height/2) * math.cos(angle) + height

    coordinates = [math.fabs(y1_new), math.fabs(x1_new), math.fabs(y2_new), math.fabs(x2_new)]

    return coordinates

    