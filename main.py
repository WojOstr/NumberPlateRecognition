import numpy as np
import cv2
import cvui
import os
from tkinter import Tk, filedialog

from numpy.lib.utils import source
import detectingletters


WINDOW_NAME = 'Number Plate Detecting App'


def load_source():
    Tk().withdraw()
    filepath = filedialog.askopenfilename(title="Load source...", filetypes=[(
        'Image Files', ['.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp']),(
        'Video Files', ['.avi','.mp4'])
        ])
    if filepath != '':
        ext = os.path.splitext(filepath)[-1].lower()
        if ext == '.mp4' or ext == '.avi':
            sourceFile = cv2.VideoCapture(filepath)
            return sourceFile, True

        else:
            sourceFile = cv2.imread(filepath, cv2.COLOR_BGRA2BGR)
            return sourceFile, None
    return None, None

def export_results(detection_array):
    for row in detection_array:
        filename = ''.join(str(e) for e in row[0])
        img = row[1]
        cv2.imwrite(".\\Results\\{}.png".format(filename), img)

def main():
    frame = np.zeros((710, 1047, 3), np.uint8)
    model = detectingletters.load_model()
    detection_array = []
    last_array_len = 0
    sourceBackground = None
    isVideoFormat = False
    count = 0
    cvui.init(WINDOW_NAME)

    while (True):

        frame[:] = (49, 52, 49)

        cvui.window(frame, 5, 5, 192, 65, 'Options')
        
        if cvui.button(frame, 20, 35, 'Open'):
            if isVideoFormat:
                sourceBackground.release()
            count = 0
            sourceBackground, isVideoFormat = load_source()
            if sourceBackground is not None:
                if isVideoFormat is None:
                    fr = sourceBackground.copy()
                    _ , detections = detectingletters.detecting(sourceBackground, model)
                    if any(detections['detection_scores'] > 0.8):
                        text, region = detectingletters.ocr_it(sourceBackground, detections, 0.6)
                        for i in range(0, len(region)):
                            if bool(text[i]):
                                temp_text = ''
                                for txt in reversed(text[i]):
                                    temp_text += txt
                                if len(temp_text) >= 4:
                                    if any(temp_text in sl for sl in detection_array):
                                        pass
                                    else:
                                        detection_array.append([temp_text, region[i]])
        if isVideoFormat:
            ret, fr = sourceBackground.read()
            if ret == True:        

                _ , detections = detectingletters.detecting(fr, model)
                if any(detections['detection_scores'] > 0.3):
                    text, region = detectingletters.ocr_it(fr, detections, 0.3)
                    for i in range(0, len(region)):
                        if bool(text[i]):
                            temp_text = ''
                            for txt in reversed(text[i]):
                                temp_text += txt
                            if len(temp_text) >= 4:
                                if any(temp_text in sl for sl in detection_array):
                                    pass
                                else:
                                    detection_array.append([temp_text, region[i]])
                count += 2 
                sourceBackground.set(cv2.CAP_PROP_POS_FRAMES, count)
            else:
                fr = np.zeros((670, 825, 3), np.uint8)
                fr[:] = (49, 52, 49)
                sourceBackground.release()
        
        if cvui.button(frame, 110, 35, 'Export'):
            export_results(detection_array)
            
        cvui.window(frame, 5, 75, 192, 630, 'Detected Numbers')

        if len(detection_array) != last_array_len:
            last_array_len = len(detection_array)
            cvui.rect(frame, 6, 90, 354, 609, 0x212121, 0x212121)

        if len(detection_array) > 6:
            start_loop = len(detection_array) - 6
        else:
            start_loop = 0

        for i in range(start_loop,len(detection_array)):
            temp_text = ''
            cvui.image(frame, 15,  100 * (i + 1 - start_loop), cv2.resize(detection_array[i][1], (172, 80)))
            for txt in detection_array[i][0]:
                temp_text += txt
            detection_array[i][0] = temp_text
            cvui.text(frame, 15, 180 * (i + 1 - start_loop) - (i - start_loop) * 80, detection_array[i][0])

            
            
        cvui.window(frame, 207, 5, 835, 700, 'Source')

        if sourceBackground is not None:
            cvui.image(frame, 212, 32, cv2.resize(fr,(825, 670)))


        cvui.update()

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(20) == 27:
            break


if __name__ == '__main__':
    main()
