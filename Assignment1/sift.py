import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from itertools import islice
import argparse
import json
parser=argparse.ArgumentParser()
parser.add_argument("--query",help="takes the query file")
parser.add_argument("--train",help="takes the train directory")
parser.add_argument("--mcnt",help="minimum number of features to be matched")
parser.add_argument("--k",help="number of ranked images to be showed")
parser.add_argument("--outfile",help="output file")
parser.add_argument("--show",action="store_true",help="show the matches")
parser.add_argument("--single",help="if match individual images")


args=parser.parse_args()
if(type(args.train)!=str and args.single==False):
    # print("hi")
    print("nothing to query into")
    exit(0)
if(args.train and args.show==True):
    print("__Hey it will show multiple images__")
    exit(0)
MIN_MATCH_COUNT=int(args.mcnt)
queryImage=args.query

if(type(args.train)==str):
    traindir=args.train
    k=int(args.k)
    img1=cv2.imread(queryImage)
    print(queryImage)
    outfile=args.outfile
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    match_dict={}
    def get_matches(trainImage,folder,match_dict):
        trainImg=traindir+folder+"/"+trainImage
        # print(trainImg)
        img2=cv2.imread(trainImg)
        gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift=cv2.xfeatures2d.SIFT_create()
        kp1,des1=sift.detectAndCompute(gray1,None)
        kp2,des2=sift.detectAndCompute(gray2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        good=[]
        for (m,n) in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        key=folder+"_"+trainImage
        match_dict[key]=len(good)
    formed_dict={}
    folders = os.listdir(traindir)
    for folder in folders:
        temp_dict={}
        files=os.listdir(traindir+folder+"/")
        start=time.time()
        for file in files:
            get_matches(file,folder,temp_dict)
            formed_dict.update(temp_dict)
        # cnt=0
        # for key in temp_dict.keys():
        #     cnt=cnt+temp_dict[key]
        print(folder)
        # print(cnt)
        end=time.time()
        print(end-start)
    sorted_dict = dict(sorted(formed_dict.items(), key=lambda kv: kv[1],reverse=True))

    exDict = {'outdict': sorted_dict}

    with open('file.txt', 'w') as file:
         file.write(json.dumps(exDict))

    with open(outfile, 'w') as f:
        for i,key in enumerate(sorted_dict.keys()):
            if(i<k):
                print(key)

            f.write(key+'\n')
else:
    trainImage=args.single
    img1=cv2.imread(queryImage)
    img2=cv2.imread(trainImage)
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(gray1,None)
    kp2,des2=sift.detectAndCompute(gray2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # ratio test as per Lowe's paper
    good=[]
    for (m,n) in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))
    if(args.show):
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            # print(img1.shape)
            h,w = gray1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            gray2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(gray1,kp1,gray2,kp2,good,None,**draw_params)

        plt.imshow(img3, 'gray'),plt.show()

