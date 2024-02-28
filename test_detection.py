from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ensemble_boxes import *
import ultralytics


image_path = 'Object_detection\\total_ds_test_imgs_yolo\\0-19824.JPEG'

model1 = YOLO("runs\\detect\\ship_detection_yoloV8m3\\weights\\last.pt")
model2 = YOLO("runs\\detect\\ship_detection_yoloV8n\\weights\\best.pt")

# results1 = model1.predict(image_path, save_txt=True)
# results2 = model1.predict(image_path, save_txt=True)


if __name__ == '__main__': 
    # print(metrics)
    # Validate the model
    metrics1 = model1.val()  # evaluate model performance on the validation set
    print(metrics1.box.map,    # map50-95
    metrics1.box.map50,  # map50
    metrics1.box.map75,  # map75
    metrics1.box.maps   # a list contains map50-95 of each category
    )
    metrics2 = model2.val()  # no arguments needed, dataset and settings remembered
    print(metrics2.box.map,    # map50-95
    metrics2.box.map50,  # map50
    metrics2.box.map75,  # map75
    metrics2.box.maps   # a list contains map50-95 of each category
    )


# from ultralytics.utils.benchmarks import benchmark

# # Benchmark on GPU
# benchmark(model='runs\\detect\\ship_detection_yoloV8n\\weights\\best.pt', data='datatest.yaml', imgsz=640, half=False, device=0)
