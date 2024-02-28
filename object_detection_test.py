from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import numpy as np


# Load a model
model = YOLO("runs\detect\yolov8_OD11\weights\\best.pt")

results = model('test images\ScanImage_BackFilter.jpg')
boxes = results[0].boxes
# print(len(boxes), boxes)
res_plotted = results[0].plot()
# print(res_plotted.shape)
# width=np.shape(res_plotted)
siz=(int(res_plotted.shape[1]/3), int(res_plotted.shape[0]/3))
source = cv2.resize(src=res_plotted, dsize=siz)
# cv2.imshow("plt", source)
# cv2.waitKey(0)

# Display the augmented image and label
plt.imshow(res_plotted)
plt.title('Augmented Label: {}'.format(results[0].probs) )
plt.show()
