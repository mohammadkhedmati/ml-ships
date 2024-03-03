python train_dual.py --device 0 --batch 16 --epochs 100 --data Object_detection/data.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights yolov9-c.pt --name yolov9-c-custom --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# train yolov9 models
python train_dual.py --workers 8 --device 0 --batch 8 --data Object_detection/data.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights yolov9-c.pt --name yolov9-c-ships --hyp hyp.scratch-high.yaml --min-items 0 --epochs 3 --close-mosaic 15

# train gelan models
# python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

python train.py --batch 16 --epochs 20 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data Object_detection/data.yaml --weights yolov9-c.pt --name yolov9-c-ships --cfg models/detect/yolov9-c.yaml --hyp hyp.scratch-high.yaml