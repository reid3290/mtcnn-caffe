# -*- coding: utf-8 -*-
from __future__ import division
import glob
import cv2
import math

def check_transformed_cords(count):
    idx = 0
    for file in glob.glob("./data/CelebA-norm-faces/*.jpg"):
        idx += 1
        img = cv2.imread(file)
        h,w,ch = img.shape
        img_name = file.split('/')[-1]
        with open("./celebA_align_transform_landmarks.txt") as anno:
            for line in anno:
                words = line.split(' ')
                if words[0] == img_name:
                    words[1:] = map(lambda x: int(x), words[1:])
                    for i in range(0,5):
                        cv2.circle( img, (words[2*i+1],words[2*i+2]), 2, (255,0,0), cv2.FILLED, cv2.LINE_AA, 0 )
                    cv2.imshow("CHECKING",img)
                    cv2.waitKey(0)
                    break
        if idx > count:
            break
# check_transformed_cords(5)
def calc_mean_landmarks():
    mean_landmarks = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    count = 0
    for file in glob.glob("./data/CelebA-norm-faces/*.jpg"):
        img = cv2.imread(file)
        h,w,ch = img.shape
        img_name = file.split('/')[-1]
        with open("./celebA_align_transform_landmarks.txt") as anno:
            for line in anno:
                words = line.split(' ')
                if words[0] == img_name:
                    words[1:] = map(lambda x: int(x), words[1:])
                    for i in range(0,5):
                        x = words[2*i+1] / w
                        y = words[2*i+2] / h
                        mean_landmarks[i][0] = mean_landmarks[i][0] + x
                        mean_landmarks[i][1] = mean_landmarks[i][1] + y
                    count += 1
                    break
    mean_landmarks = map(lambda x: (x[0]/count, x[1]/count), mean_landmarks)
    print mean_landmarks
# calc_mean_landmarks()
mean_landmarks = [
    (0.2644648557981477, 0.3893615695621327), 
    (0.7298884742287275, 0.38820151858624363), 
    (0.4948865054954475, 0.5950776954681372), 
    (0.2837772174882209, 0.7280642484406197), 
    (0.7076527476746144, 0.729596680867603)]

def check_mean_landmarks(path):
    count = 0
    for img_file in glob.glob(path):
        print img_file
        img = cv2.imread(img_file)
        # img = cv2.resize(img, (size,size))
        h,w,ch = img.shape
        for i in range(0, 5):
            x = int(mean_landmarks[i][0] * w)
            y = int(mean_landmarks[i][1] * h)
            cv2.circle( img, (x,y), 2, (255,0,0), cv2.FILLED, cv2.LINE_AA, 0 )
        cv2.imshow("CHECKING",img)
        cv2.waitKey(0)
        count += 1
        if count > 5:
            break
# check_mean_landmarks("./data/CelebA-norm-faces/*.jpg")
# check_mean_landmarks("./data/CelebA 脸/*.jpg")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def score(p1, p2):
    deltaX = p1[0] - p2[0]
    deltaY = p1[1] - p2[1]
    delta = math.sqrt(deltaX**2 + deltaY**2) * 10
    # print delta
    if delta < 1:
        return 1
    else:
        return 1/delta
    # return sigmoid(1/delta)
def check_landmark_scores():
    idx = 0
    for file in glob.glob("./data/CelebA-faces/*.jpg"):
        idx += 1
        img = cv2.imread(file)
        h,w,ch = img.shape
        img_name = file.split('/')[-1]
        with open("./celebA_align_transform_landmarks.txt") as anno:
            for line in anno:
                words = line.split(' ')
                if words[0] == img_name:
                    words[1:] = map(lambda x: int(x), words[1:])
                    for i in range(0,5):
                        s = score(mean_landmarks[i], (words[2*i+1]/w,words[2*i+2]/h))
                        print s
                        cv2.putText(img,str(round(s,1)).split('.')[1], (words[2*i+1],words[2*i+2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                    print '*'*10
                    cv2.imshow("CHECKING",img)
                    cv2.waitKey(0)
                    break
        if idx > 30:
            break

def calc_landmark_scores(image_set):
    count = 0
    with open("./%s_transform_landmarks_scores.txt" % image_set,'w') as target:
        with open("./%s_transform_landmarks.txt" % image_set) as anno:
            for line in anno:
                words = line.split(' ')
                img = cv2.imread("./%s_face/" % image_set + words[0])
                h,w,ch = img.shape
                words[1:] = map(lambda x: int(x), words[1:])
                for i in range(0,5):
                    s = score(mean_landmarks[i], (words[2*i+1]/w,words[2*i+2]/h))
                    words.append(s)
                words = map(lambda x: str(x), words)
                target.write(' '.join(words)+"\n")
                count += 1
                if count % 100 == 0:
                    print count
# check_landmark_scores()
# calc_landmark_scores('celebA')
calc_landmark_scores('83k')