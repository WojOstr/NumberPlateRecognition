import cv2
import uuid
import os

labels = ['bigplate', 'smallplate']
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not(os.path.exists(IMAGES_PATH)):
    os.mkdir(IMAGES_PATH)

for label in labels:
    DIR = './simpleimages/{}'.format(label)

    for img in os.listdir(DIR):
        image = cv2.imread(os.path.join(DIR, img))
        print(image)
        if image is not None:
            imgname = os.path.join(".\\"+IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imgname, image)

