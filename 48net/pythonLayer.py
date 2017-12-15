import sys
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 64
        net_side = 48
        pts_list = ''
        scr_list = ''
        pts_root = ''
        scr_root = ''
        self.batch_loader = BatchLoader(pts_list,scr_list,net_side,pts_root,scr_root)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 10)
        top[2].reshape(self.batch_size, 5)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        loss_task = random.randint(0,1)
        #loss_task = 1
        for itt in range(self.batch_size):
            im, pts, scr = self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = pts
            top[2].data[itt, ...] = scr
    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,pts_list,scr_list,net_side,pts_root,scr_root):
        self.mean = 128
        self.im_shape = net_side
        self.pts_root = pts_root
        self.pts_list = []
        self.scr_root = scr_root
        self.scr_list = []

        print "Start Reading pts-regression Data into Memory..."
        fid = open('48/pts_train.imdb','r')
        self.pts_list = pickle.load(fid)
        fid.close()
        random.shuffle(self.pts_list)
        self.pts_cur = 0
        print "\n",str(len(self.pts_list))," pts-regression Data have been read into Memory..."

        print "Start Reading scr-regression Data into Memory..."
        fid = open('48/scr_train.imdb','r')
        self.scr_list = pickle.load(fid)
        fid.close()
        random.shuffle(self.scr_list)
        self.scr_cur = 0
        print "\n",str(len(self.scr_list))," scr-regression Data have been read into Memory..."


    def load_next_image(self,loss_task):
        if loss_task == 0:
            if self.pts_cur == len(self.pts_list):
                self.pts_cur = 0
                random.shuffle(self.pts_list)
            cur_data = self.pts_list[self.pts_cur]  # Get the image index
            im = cur_data[0]
            pts = cur_data[1]
            #print "points,", pts
            scr = [-1, -1, -1, -1, -1]
            self.pts_cur += 1
            return im, pts, scr
        elif loss_task == 1:
            if self.scr_cur == len(self.scr_list):
                self.scr_cur = 0
                random.shuffle(self.scr_list)
            cur_data = self.scr_list[self.scr_cur]  # Get the image index
            im = cur_data[0]
            pts = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            scr = cur_data[1]
            self.scr_cur += 1
            return im, pts, scr
################################################################################
######################Regression Loss Layer By Python###########################
################################################################################
class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        pts = bottom[1].data
        self.valid_index = np.where(pts[:,0] != -1)[0]
        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i] or self.N==0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num