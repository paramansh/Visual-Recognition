import cv2
import os
path1 = "/home/paramansh/VR/Visual-Recognition/Assignment1/train/"
folders = os.listdir(path1)
for folder in folders:
    # print(folder)
    path="data/"+str(folder)
    if not os.path.exists(path):
            os.mkdir(path)
    # os.mkdir(path)
    # exit(0)
    path2 = path1+str(folder)+"/"
    files = os.listdir(path2)
    # print(files)
    # exit(0)
    for file in files:
        filename=path2+str(file)
        # print(filename)
        img=cv2.imread(filename)
        # print(img.shape)
        # exit(0)
        crop_img = img[50:350,200:400]
        outfile = path+"/"+str(file)
        cv2.imwrite(outfile,crop_img)
