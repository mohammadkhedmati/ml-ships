from ultralytics import YOLO
from matplotlib import pyplot as plt

def train_yolo():

    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8m.yaml")  # build a new model from scratch

    # load a pretrained model (recommended for training)
    
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolo_base_weights\\yolov8m.pt")
    # model = YOLO("runs\\detect\\ship_detection_yoloV8n\\weights\\last.pt")


    # Training
    results = model.train(
        data='H:\\Code\\Python\\ML-ship\\data.yaml',
        epochs=150,
        batch=16,
        name='ship_detection_yoloV8m',
        cache=True,
    )

    metrics = model.val()  # evaluate model performance on the validation set
    print(metrics)

    return results

    # success = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    result = train_yolo()
    print(result)
