
import os
import sys
import cv2
import h5py
import numpy as np

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from SphereProjection import SphereProjection
from util.network import load_network, forward_patch, load_sphconv, forward_sphconv
from util.rf import kernel_sizes, strides

class SphConv(object):
    def __init__(self, layer, sphereH=320, ks=640, fov=65.5):
        sphereW = sphereH *2
        base = "1_1"
        kernel_size = kernel_sizes[base]
        projection = SphereProjection(kernel_size=kernel_size,
                                      sphereW=sphereW,
                                      sphereH=sphereH,
                                      view_angle=fov,
                                      imgW=ks)
        base_net = load_network(layer=base, network='faster-rcnn')
        n_c = base_net.blobs['conv{}'.format(base)].shape[1]
        stride = strides[ks][sphereH][layer]

        self.sphereW = sphereW
        self.sphereH = sphereH
        self.projection = projection
        self.base_net = base_net
        self.base = base
        self.n_c = n_c
        self.Ps = {}
        self.stride = stride

    def extract_convs(self, img, xs, ys, base_path=None, conv_path=None):
        if conv_path is not None:
            frameId = os.path.splitext(os.path.basename(conv_path))[0]
            with h5py.File(conv_path, 'r') as hf:
                vals = hf[frameId][:]
        else:
            if img.shape[0] != self.sphereH:
                img = cv2.resize(img, (self.sphereW, self.sphereH))
            
            if base_path is None:
                base = self.extract_base(img)
            else:
                frameId = os.path.splitext(os.path.basename(base_path))[0]
                with h5py.File(base_path, 'r') as hf:
                    base = hf[frameId][:]
            vals = forward_sphconv(base, self.net)

        result = []
        for x, y in zip(xs, ys):
            x /= self.stride
            y /= self.stride
            val = vals[y,x,:]
            result.append(val)
        return result

    def extract_base(self, img):
        base = np.zeros((self.sphereH, self.sphereW, self.n_c))
        for i in xrange(self.sphereH):
            if i in self.Ps:
                P = self.Ps[i]
            else:
                P = self.projection.buildP(i)
                self.Ps[i] = P
            for j in xrange(self.sphereW):
                rimg = np.roll(img, self.sphereW/2-j, axis=1)
                patch = self.projection.project(P, rimg)
                val = forward_patch(patch, self.base_net, layer=self.base)
                base[i,j,:] = val.ravel()
        return base
