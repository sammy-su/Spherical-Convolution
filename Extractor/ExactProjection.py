
import os
import sys
import cv2
import numpy as np

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from SphereProjection import SphereProjection
from util.network import load_network, forward_patch
from util.rf import kernel_sizes

class ExactProjection(object):
    def __init__(self, layer, sphereH=320, ks=640, fov=65.5):
        sphereW = sphereH * 2
        kernel_size = kernel_sizes[layer]
        projection = SphereProjection(kernel_size=kernel_size,
                                      sphereW=sphereW,
                                      sphereH=sphereH,
                                      view_angle=fov,
                                      imgW=ks)
        net = load_network(layer=layer, network="faster-rcnn")

        self.sphereH = sphereH
        self.sphereW = sphereW
        self.Ps = {}
        self.projection = projection
        self.layer = layer
        self.net = net

    def extract_conv(self, img, x, y):
        if y in self.Ps:
            P = self.Ps[y]
        else:
            P = self.projection.buildP(y)
            self.Ps[y] = P

        if img.shape[0] != self.sphereH:
            img = cv2.resize(img, (self.sphereW, self.sphereH))
        rimg = np.roll(img, self.projection.sphereW/2-x, axis=1)
        patch = self.projection.project(P, rimg)
        val = forward_patch(patch, self.net, layer=self.layer)
        return val.ravel()

    def extract_convs(self, img, xs, ys):
        if img.shape[0] != self.sphereH:
            img = cv2.resize(img, (self.sphereW, self.sphereH))

        vals = []
        for x, y in zip(xs, ys):
            val = self.extract_conv(img, x, y)
            vals.append(val)
        return vals

