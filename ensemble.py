from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
from ensemble_boxes import *


image_path = 'Object_detection\\total_ds_test_imgs_yolo\\0-19824.JPEG'

model1 = YOLO("runs\\detect\\ship_detection_yoloV8m3\\weights\\best.pt")
model2 = YOLO("runs\\detect\\ship_detection_yoloV8n\\weights\\best.pt")

results1 = model1.predict(image_path, save_txt=True)
results2 = model1.predict(image_path, save_txt=True)

boxes1 = results1[0].boxes
boxes2 = results2[0].boxes

# res_plotted = results[0].plot()

# Display the augmented image and label
# plt.imshow(res_plotted)
# plt.title('Augmented Label: {}'.format(results[0].probs) )
# plt.show()
labels1 =[]
for label in boxes1.cls.tolist():
    labels1.append(int(label))

labels2 =[]
for label in boxes2.cls.tolist():
    labels2.append(int(label))

boxes_list = [boxes1.xyxyn.tolist(),boxes2.xyxyn.tolist()]
scores_list = [boxes1.conf.tolist(), boxes2.conf.tolist()]
labels_list = [labels1, labels2]
weights = [2, 1]

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
print('nms \n', boxes, scores, labels)

boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
print('soft_nms \n', boxes, scores, labels)

boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
print('non_maximum_weighted \n', boxes, scores, labels)

boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
print('weighted_boxes_fusion \n', boxes, scores, labels)

