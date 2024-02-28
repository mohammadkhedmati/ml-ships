from PIL import Image
import os, sys

path = "H:\\Code\\Python\\ML-ship\\Object_detection\\test\\images\\"
save_path = "H:\\Code\\Python\\ML-ship\\Object_detection\\test\\resized_images\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            if im.size[0] % 32 == 0:
                continue
            f, e = os.path.splitext(path+item)
            name = item.split('.')[0]
            final_path = save_path + name
            if im.size[0] < 1024 :
                x = im.size[0] / 512
                height = im.size[1]/x
                imResize = im.resize((512, int(height)))
                imResize.save(final_path + '.jpg', 'JPEG', quality=100)
            x = im.size[0] / 1024
            height = im.size[1]/x
            imResize = im.resize((1024, int(height)))

resize()