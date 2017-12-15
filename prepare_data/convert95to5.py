# -*- coding: utf-8 -*-
import cv2
import glob
image_dir = "/Users/reid/83k"

def parse_annotation(line):
    line = line.split("\t")
    img_name = line[0]
    img_annos = line[1]
    points = map(lambda x, y: (int(float(x)),int(float(y))), line[1].split(",")[::2], line[1].split(",")[1::2]) 
    return img_name, points

# 把 95 个关键点的人脸数据转换成 5 个关键点的
def convert95to5(sub_dir):
    new_anno = open(image_dir+"/" + sub_dir + "-5points.txt", "w")
    with open(image_dir + "/" + sub_dir + ".txt", "r") as annotations:
        for line in annotations:
            img_name, points = parse_annotation(line)
            points = [points[i] for i in [0,9,34,46,47]]

            new_line = sub_dir + "/" + img_name + "\t"
            for p in points:
                new_line += str(p[0]) + "," + str(p[1]) + ","
            new_line = new_line[:-1]
            new_line += "\n"
            new_anno.write(new_line)

# 画出人脸的 5 个关键点看看对不对
def checkKeypoints(sub_dir, num=5):
    count = 0
    with open(image_dir+"/" + sub_dir + "-5points.txt", "r") as annotations:
        for line in annotations:
            img_name, points = parse_annotation(line)
            img_path = image_dir+"/"+img_name
            img = cv2.imread(img_path)
            if img is None:
                print "read image failed, ", img_path
                return
            for p in points:
                cv2.circle( img, p, 2, (255,0,0), cv2.FILLED, cv2.LINE_AA, 0 )
            cv2.imshow("CHECKING",img)
            cv2.waitKey(0)
            count += 1
            if count > num:
                break
# checkKeypoints("pic1")
