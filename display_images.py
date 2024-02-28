import matplotlib.pyplot as plt
import cv2 
import os
# source data
image_path = "H:\\Code\\Python\\ML-ship\\Object_detection\\train\\images\\"
label_path = "H:\\Code\\Python\\ML-ship\\Object_detection\\train\\labels\\"
save_image_path = "H:\\Code\\Python\\ML-ship\\Object_detection\\train\\resized_images\\"
save_label_path = "H:\\Code\\Python\\ML-ship\\Object_detection\\train\\resized_label\\"
dirs = os.listdir( image_path )

def plot_image(image_path, label_path):
    # create an OpenCV image
    img = cv2.imread(image_path)
    try:
        height, width, channels = img.shape
    except:
        print('no shape info.')

    file1 = open(label_path, 'r')
    Lines = file1.readlines()

    for line in Lines: 
        staff = line.split()
        class_idx = int(staff[0])

        x_center, y_center, w, h = float(
            staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)

        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        image = cv2.rectangle(img, c1, c2, (0,255,0), 2)

    cv2.imshow('window_name', image)   
    cv2.waitKey(0)

def resize_image(image_path, label_path, file_name):
    im = cv2.imread(image_path)
    orginal_height, orginal_width, channels = im.shape

    if im.shape[1] % 32 == 0:
        pass
    if im.shape[1] < 1024 :
        width = im.shape[1] / 512
        height = im.shape[0]/ width
        up_points = (512, round(height))
        resized_up = cv2.resize(im, up_points)
        file1 = open(label_path, 'r')
        Lines = file1.readlines()

        for line in Lines: 
            staff = line.split()
            class_idx = int(staff[0])

            x_center, y_center, w, h = float(staff[1])*orginal_width,float(staff[2])*orginal_height,float(staff[3])*orginal_width,float(staff[4])*orginal_height
            ratio = [orginal_width / 512 , orginal_height / height]
            x_center, y_center, w, h = x_center/ratio[0] , y_center/ratio[1], w/ratio[0], h/ratio[1]
            yolo_x_center, yolo_y_center, yolo_w, yolo_h = x_center/1024 , y_center/round(height), w/1024, h/round(height)

            # x1 = round(x_center-w/2)
            # y1 = round(y_center-h/2)
            # x2 = round(x_center+w/2)
            # y2 = round(y_center+h/2)

            # c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            # resized_up_label = cv2.rectangle(resized_up, c1, c2, (0,255,0), 2)
            save_label = save_label_path + file_name + '.txt'
            save_image = save_image_path + file_name + '.JPEG'
            f= open(save_label,"a+")
            f.write('{} {} {} {} {} \n'.format(class_idx,yolo_x_center, yolo_y_center, yolo_w, yolo_h))
            cv2.imwrite(save_image, resized_up) 

    if im.shape[1] > 1024 :
        width = im.shape[1] / 1024
        height = im.shape[0]/width
        up_points = (1024, round(height))
        resized_up = cv2.resize(im, up_points)
        file1 = open(label_path, 'r')
        Lines = file1.readlines()
        for line in Lines: 
            staff = line.split()
            class_idx = int(staff[0])

            x_center, y_center, w, h = float(staff[1])*orginal_width,float(staff[2])*orginal_height,float(staff[3])*orginal_width,float(staff[4])*orginal_height
            ratio = [orginal_width / 1024 , orginal_height / height]
            x_center, y_center, w, h = x_center/ratio[0] , y_center/ratio[1], w/ratio[0], h/ratio[1]
            yolo_x_center, yolo_y_center, yolo_w, yolo_h = x_center/1024 , y_center/round(height), w/1024, h/round(height)

            # x1 = round(x_center-w/2)
            # y1 = round(y_center-h/2)
            # x2 = round(x_center+w/2)
            # y2 = round(y_center+h/2)

            # c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            # resized_up_label = cv2.rectangle(resized_up, c1, c2, (0,255,0), 2)
            save_label = save_label_path + file_name + '.txt'
            save_image = save_image_path + file_name + '.JPEG'
            f= open(save_label,"a+")
            f.write('{} {} {} {} {} \n'.format(class_idx,yolo_x_center, yolo_y_center, yolo_w, yolo_h))
            cv2.imwrite(save_image, resized_up) 

    # cv2.imshow('window_name', resized_up_label)   
    # cv2.waitKey(0)

def resize_all_images(image_path, label_path, file_name):
        im = cv2.imread(image_path)
        orginal_height, orginal_width, channels = im.shape
        # 640×480
        # width = im.shape[1] / 640
        # height = im.shape[0]/width
        up_points = (640, 480)
        resized_up = cv2.resize(im, up_points)
        file1 = open(label_path, 'r')
        Lines = file1.readlines()
        for line in Lines: 
            staff = line.split()
            class_idx = int(staff[0])

            x_center, y_center, w, h = float(staff[1])*orginal_width,float(staff[2])*orginal_height,float(staff[3])*orginal_width,float(staff[4])*orginal_height
            ratio = [orginal_width / 640 , orginal_height / 480]
            x_center, y_center, w, h = x_center/ratio[0] , y_center/ratio[1], w/ratio[0], h/ratio[1]
            yolo_x_center, yolo_y_center, yolo_w, yolo_h = x_center/640 , y_center/480, w/640, h/480

            # x1 = round(x_center-w/2)
            # y1 = round(y_center-h/2)
            # x2 = round(x_center+w/2)
            # y2 = round(y_center+h/2)

            # c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            # resized_up_label = cv2.rectangle(resized_up, c1, c2, (0,255,0), 2)
            save_label = save_label_path + file_name + '.txt'
            save_image = save_image_path + file_name + '.JPEG'
            f= open(save_label,"a+")
            f.write('{} {} {} {} {}\n'.format(class_idx,yolo_x_center, yolo_y_center, yolo_w, yolo_h))
            cv2.imwrite(save_image, resized_up) 

def resize():
    for item in dirs:
        if os.path.isfile(image_path+item):
            label_name = item.split('.')[0] + '.txt'
            file_name = item.split('.')[0]
            # resize_image(image_path=image_path+item, label_path=label_path+label_name, file_name = file_name)
            resize_all_images(image_path=image_path+item, label_path=label_path+label_name, file_name = file_name)

resize()

# plot_image(image_path="H:\\Code\\Python\\ML-ship\\Object_detection\\test\\resized_images\\0-19824.JPEG", label_path="H:\\Code\\Python\\ML-ship\\Object_detection\\test\\resized_label\\0-19824.txt")

