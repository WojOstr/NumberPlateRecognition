import cv2
import uuid
import os
from PIL import Image
import random
import numpy as np

labels = ['licence']
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
convert = True
if not(os.path.exists(IMAGES_PATH)):
    os.mkdir(IMAGES_PATH)

for label in labels:
    DIR = './simpleimages/{}'.format(label)

    for img in os.listdir(DIR):
        image = cv2.imread(os.path.join(DIR, img))
        if convert:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            size = random.randint(700,1800), random.randint(700,1800)
            im_pil.thumbnail(size)
            image = np.asarray(im_pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image is not None:
            imgname = os.path.join(".\\"+IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imgname, image)

