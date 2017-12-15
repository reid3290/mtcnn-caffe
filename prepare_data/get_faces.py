import glob
import cv2
import os
def extract_face(annotation_file, image_format):
    fid = open(annotation_file)
    lines = fid.readlines()
    img_path = lines[0].strip()
    face_num = int(lines[1].strip())
    img = cv2.imread(img_path + image_format)
    max_face_score = 0
    face = None
    for i in range(2,2 + face_num):
        face_score = float(lines[i].split(" ")[-1])
        if max_face_score < face_score:
            max_face_score = face_score
            cords = map(lambda x: int(float(x)), lines[i].split(" ")[:-1])
            left = (cords[0], cords[1])
            right = (cords[0] + cords[2], cords[1] + cords[3])
            # cv2.rectangle(img, left, right, (0,0,255))
            face = img[max(0,cords[1]):cords[1]+cords[3],max(0,cords[0]):cords[0]+cords[2]]
    return face
def get_faces(image_set_name, image_format):
    faces_dir = '/disk2/zjh/RefineDet/data/face-rd-18-t0/%s_face/' % image_set_name
    if not os.path.exists(faces_dir):
        os.mkdir(faces_dir)
    idx = 0
    for anno in glob.glob('/disk2/zjh/RefineDet/data/face-rd-18-t0/%s_output/*.txt' % image_set_name):
        face = extract_face(anno, image_format)
        img_name = anno.split('/')[-1].split(".")[0]
        cv2.imwrite( faces_dir + img_name + ".jpg", face)
        idx += 1
        if idx % 100 == 0:
                print idx

get_faces('83k','.jpg')