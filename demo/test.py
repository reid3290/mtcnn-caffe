import sys
sys.path.append('.')
import tools_matrix as tools
import caffe
import cv2
import numpy as np

deploy = '../48net/test.prototxt'
caffemodel = '../48net/48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)


def detectFace(img_path):
    img = cv2.imread(img_path)
    h,w,ch = img.shape
    if h!=48 or w!=48:
        img = cv2.resize(img,(48,48))
    img = np.swapaxes(img, 0, 2)
    img = (img - 127.5)/127.5
    net_48.blobs['data'].reshape(1,3,48,48)
    net_48.blobs['data'].data[...]=img
    caffe.set_device(0)
    caffe.set_mode_gpu()
    out = net_48.forward()

    # print img[0]
    # print out
    # print out['fc6-3'][0]
    # print out['fc6-2'][0]
    return out['fc6-3'][0], out['fc6-2'][0]

    # return np.amin(np.array(out['fc6-2'][0]))
norm_faces = [
"000007.jpg", "000031.jpg", "000083.jpg", "000133.jpg", "000221.jpg", "000357.jpg", 
"000401.jpg", "000440.jpg", "000489.jpg", "000010.jpg", "000032.jpg", "000097.jpg", 
"000152.jpg", "000239.jpg", "000359.jpg", "000402.jpg", "000442.jpg", "000492.jpg",
"000011.jpg", "000035.jpg", "000099.jpg", "000153.jpg", "000249.jpg", "000361.jpg", 
"000414.jpg", "000445.jpg", "000493.jpg", "000012.jpg", "000038.jpg", "000104.jpg", 
"000157.jpg", "000264.jpg", "000362.jpg", "000415.jpg", "000457.jpg", "000494.jpg",
"000013.jpg", "000040.jpg", "000108.jpg", "000176.jpg", "000312.jpg", "000365.jpg", 
"000416.jpg", "000461.jpg", "000495.jpg", "000014.jpg", "000042.jpg", "000109.jpg", 
"000201.jpg", "000333.jpg", "000368.jpg", "000419.jpg", "000465.jpg", "000496.jpg",
"000018.jpg", "000043.jpg", "000113.jpg", "000203.jpg", "000340.jpg", "000377.jpg", 
"000427.jpg", "000466.jpg", "000497.jpg", "000019.jpg", "000047.jpg", "000115.jpg", 
"000207.jpg", "000348.jpg", "000382.jpg", "000428.jpg", "000467.jpg", "000021.jpg", 
"000055.jpg", "000116.jpg", "000211.jpg", "000349.jpg", "000386.jpg", "000430.jpg", 
"000470.jpg", "000023.jpg", "000062.jpg", "000121.jpg", "000212.jpg", "000352.jpg", 
"000388.jpg", "000432.jpg", "000476.jpg", "000026.jpg", "000064.jpg", "000122.jpg", 
"000213.jpg", "000353.jpg", "000392.jpg", "000434.jpg", "000478.jpg", "000028.jpg", 
"000065.jpg", "000125.jpg", "000214.jpg", "000354.jpg", "000398.jpg", "000435.jpg", 
"000484.jpg", "000029.jpg", "000082.jpg", "000129.jpg", "000216.jpg", "000356.jpg", 
"000399.jpg", "000436.jpg", "000488.jpg"
]
faces = [
"000001.jpg", "000004.jpg", "000039.jpg", "000049.jpg", "000061.jpg", "000069.jpg", 
"000074.jpg", "000077.jpg", "000080.jpg", "000002.jpg", "000036.jpg", "000041.jpg", 
"000059.jpg", "000067.jpg", "000070.jpg", "000075.jpg", "000078.jpg", "000081.jpg",
"000003.jpg", "000037.jpg", "000048.jpg", "000060.jpg", "000068.jpg", "000073.jpg", 
"000076.jpg", "000079.jpg", "000084.jpg", "100001.jpg", "100002.jpg", "100003.jpg",
"100004.jpg","200001.jpg","200002.jpg","200003.jpg","200004.jpg"
]

# detectFace("/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_face/000001.jpg")
# detectFace("/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_face/000007.jpg")
fid = open("result.txt",'w')
for face in norm_faces:
    print face
    imgpath = "/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_face/" + face
    pts, scr = detectFace(imgpath)
    fid.write(face + " ")
    for x in pts:
        fid.write(str(x)+" ")
    for x in scr:
        fid.write(str(x)+" ")
    fid.write("\n")
for face in faces:
    print face
    imgpath = "/disk2/zjh/RefineDet/data/face-rd-18-t0/celebA_face/" + face
    pts, scr = detectFace(imgpath)
    fid.write(face + " ")
    for x in pts:
        fid.write(str(x)+" ")
    for x in scr:
        fid.write(str(x)+" ")
    fid.write("\n")