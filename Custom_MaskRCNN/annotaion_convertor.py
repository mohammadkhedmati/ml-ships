from pathlib import Path

import globox


path = Path("H:\\Code\\Python\\ML-ship\\Object_detection\\total_ds_val_imgs_yolo")  # Where the .txt files are
image_folder="H:\\Code\\Python\\ML-ship\\Object_detection\\total_ds_val_imgs_yolo"
save_file = Path("val.json")

annotations = globox.AnnotationSet.from_yolo_v7(folder=path, image_folder=image_folder)
annotations.save_coco(path=save_file,auto_ids=True)
