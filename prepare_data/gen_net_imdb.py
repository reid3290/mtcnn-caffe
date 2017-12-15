import sys
import cv2
import os
import numpy as np
import numpy.random as npr
import cPickle as pickle
import random

size = 48
net = str(size)
pts_list = []
scr_list = []
    
    
def read_image(path):
    im = cv2.imread(path)
    try:
        h,w,ch = im.shape
    except:
        print "%s does not exist" % path
        return
    if h!=48 or w!=48:
        im = cv2.resize(im,(48,48))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    
    pts = [ float(words[1])/w,float(words[2])/h,
            float(words[3])/w,float(words[4])/h,
            float(words[5])/w,float(words[6])/h,
            float(words[7])/w,float(words[8])/h,
            float(words[9])/w,float(words[10])/h ]
    scr = [ float(words[11]),float(words[12]),float(words[13]),float(words[14]),float(words[15]) ]
    pts_list.append([im,pts])
    scr_list.append([im,scr])

print "Reading CelebA faces..."
with open('/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_align_transform_landmarks_scores.txt', 'r') as f:
    landmarks = f.readlines()
cur_ = 0
sum_ = len(landmarks)
for line in landmarks:
    cur_  += 1
    if cur_ % 1000 == 0:
        print cur_

    words = line.split()
    image_file_path = '/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_face/'+words[0]

    read_image(image_file_path)
print "CelebA face number: %d" % cur_

print "Reading 83k faces..."
with open('/disk2/zjh/RefineDet/data/face-rd-18-t0/83k_transform_landmarks_scores.txt', 'r') as f:
    landmarks = f.readlines()
cur_ = 0
sum_ = len(landmarks)
for line in landmarks:
    cur_  += 1
    if cur_ % 1000 == 0:
        print cur_

    words = line.split()
    image_file_path = '/disk2/zjh/RefineDet/data/face-rd-18-t0/83k_face/'+words[0]

    read_image(image_file_path)
print "83k face number: %d" % cur_

total = len(pts_list)
l1 = int( 0.05 * total )
l2 = int( 0.1 * total )

random.shuffle(pts_list)
random.shuffle(scr_list)

fid = open("../48net/48/pts_dev.imdb",'w')
pickle.dump(pts_list[:l1], fid)
fid.close()
fid = open("../48net/48/scr_dev.imdb",'w')
pickle.dump(scr_list[:l1], fid)
fid.close()

fid = open("../48net/48/pts_test.imdb",'w')
pickle.dump(pts_list[l1:l2], fid)
fid.close()
fid = open("../48net/48/scr_test.imdb",'w')
pickle.dump(scr_list[l1:l2], fid)
fid.close()

fid = open("../48net/48/pts_train.imdb",'w')
pickle.dump(pts_list[l2:], fid)
fid.close()
fid = open("../48net/48/scr_train.imdb",'w')
pickle.dump(scr_list[l2:], fid)
fid.close()
