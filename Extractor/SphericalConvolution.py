
import os
import sys
import cv2
import numpy as np

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from SphereProjection import SphereProjection
from util.network import load_network, forward_patch, use_BatchNorm
from util.rf import kernel_sizes, top_down, strides, rounded_rf

class SphericalConvolution(object):
    def __init__(self, layer, sphereH=320, ks=640, fov=65.5):
        sphereW = sphereH *2
        bot = top_down[layer]
        kernel_size = kernel_sizes[bot]
        projection = SphereProjection(kernel_size=kernel_size,
                                      sphereW=sphereW,
                                      sphereH=sphereH,
                                      view_angle=fov,
                                      imgW=ks)
        base_net = load_network(layer=bot, network='faster-rcnn')
        n_c = base_net.blobs['conv{}'.format(bot)].shape[1]

        supports = {}
        stride = strides[ks][sphereH][layer]
        for y in xrange(sphereH):
            crop_in, _ = rounded_rf(layer, y, sphereH, ks)
            h, w = crop_in
            n_w = (w-1) / stride / 2
            n_h = (h-1) / stride / 2

            hs = np.arange(-n_h*stride, n_h*stride+1, step=stride)
            ws = np.arange(-n_w*stride, n_w*stride+1, step=stride)
            Px, Py = np.meshgrid(hs, ws)
            Py += y
            supports[y] = (Px, Py)
        self.supports = supports
        self.n_c = n_c

        self.sphereW = sphereW
        self.sphereH = sphereH
        self.ks = ks
        self.layer = layer
        self.bot = bot
        self.base_net = base_net
        self.projection = projection
        self.Ps = {}
        self.top_nets = {}

    def extract_convs(self, img, xs, ys):
        if img.shape[0] != self.sphereH:
            img = cv2.resize(img, (self.sphereW, self.sphereH))

        bases = self._construct_bases(xs, ys)
        base_vals = self.extract_bases(img, bases)
        vals = []
        for x, y in zip(xs, ys):
            x = 2 * (x / 2)
            y = 2 * (y / 2)
            val = self.extract_conv(img, x, y, base_vals=base_vals)
            vals.append(val)
        return vals

    def _construct_bases(self, xs, ys):
        bases = set()
        for x, y in zip(xs, ys):
            x = 2 * (x / 2)
            y = 2 * (y / 2)
            base = self._construct_base(x, y)
            bases.update(base)
        return bases

    def _construct_base(self, x, y):
        bases = set()
        Px, Py = self.supports[y]
        for i in xrange(Px.shape[0]):
            for j in xrange(Px.shape[1]):
                col = Px[i,j] + x
                row = Py[i,j]
                col, row = self.translate_coordinate(col, row)
                key = (col, row)
                bases.add(key)
        return bases

    def extract_bases(self, img, bases):
        if self.sphereH != img.shape[0]:
            img = cv2.resize(img, (self.sphereW, self.sphereH))

        batch_size = 128
        in_shape = self.base_net.blobs['data'].shape
        self.base_net.blobs['data'].reshape(batch_size, in_shape[1], in_shape[2], in_shape[3])
        self.base_net.reshape()

        bidx = 0
        vals = []
        if "_" in self.bot:
            target = 'conv{}'.format(self.bot)
        else:
            target = 'fc{}'.format(self.bot)
        mean = np.array([103.939, 116.779, 123.68])

        for x, y in bases:
            if y in self.Ps:
                P = self.Ps[y]
            else:
                P = self.projection.buildP(y)
                self.Ps[y] = P
            rimg = np.roll(img, self.sphereW/2-x, axis=1)
            patch = self.projection.project(P, rimg)
            patch = patch - mean
            patch = np.transpose(patch, (2,0,1))
            self.base_net.blobs['data'].data[bidx,...] = patch
            bidx += 1
            if bidx == batch_size:
                bidx = 0
                out = self.base_net.forward(blobs=[target])
                val = out[target].copy()
                for i in xrange(batch_size):
                    vals.append(val[i].ravel())
        if bidx > 0:
            out = self.base_net.forward(blobs=[target])
            val = out[target].copy()
            for i in xrange(bidx):
                vals.append(val[i].ravel())

        base_vals = {}
        for key, val in zip(bases, vals):
            base_vals[key] = val

        self.base_net.blobs['data'].reshape(in_shape[0], in_shape[1], in_shape[2], in_shape[3])
        self.base_net.reshape()
        return base_vals

    def extract_conv(self, img, x, y, base_vals=None):
        if self.sphereH != img.shape[0]:
            img = cv2.resize(img, (self.sphereW, self.sphereH))

        Px, Py = self.supports[y]
        data = np.zeros((Px.shape[0], Px.shape[1], self.n_c))
        for i in xrange(Px.shape[0]):
            for j in xrange(Px.shape[1]):
                col = Px[i,j] + x
                row = Py[i,j]
                col, row = self.translate_coordinate(col, row)
                key = (col, row)
                if isinstance(base_vals, dict) and key in base_vals:
                    val = base_vals[key]
                else:
                    val = self.extract_base(img, col, row)
                    base_vals[key] = val
                data[i,j,:] = val
        INDn = data < 0
        data[INDn] = 0.
        data = np.transpose(data, (2,0,1))

        if y in self.top_nets:
            top_net = self.top_nets[y]
        else:
            top_net = self.load_network(y)
            self.top_nets[y] = top_net
        top_net.blobs['data'].data[0] = data
        target = 'conv{0}/{1}'.format(self.layer, y)
        out = top_net.forward(blobs=[target])
        val = out[target].copy()
        return val.ravel()

    def translate_coordinate(self, x, y):
        if y < 0:
            y = -1 - y
            x += self.sphereW / 2
        elif y >= self.sphereH:
            y = self.sphereH - 1 - (y - self.sphereH)
            x += self.sphereW / 2
        if x >= self.sphereW:
            x -= self.sphereW
        elif x < 0:
            x += self.sphereW
        return x, y

    def extract_base(self, img, x, y):
        if y in self.Ps:
            P = self.Ps[y]
        else:
            P = self.projection.buildP(y)
            self.Ps[y] = P
        rimg = np.roll(img, self.sphereW/2-x, axis=1)
        patch = self.projection.project(P, rimg)
        val = forward_patch(patch, self.base_net, layer=self.bot)
        return val.ravel()

    def load_network(self, y, silence=True):
        if use_BatchNorm(self.layer):
            itr_fin = 4000
        else:
            itr_fin = 16000
        proto = os.path.join(SphConv_ROOT,
                             "prototxt",
                             "SphereH{0}Ks{1}".format(self.sphereH, self.ks),
                             "faster-rcnn{}".format(self.layer),
                             "deploy.tilt{:03d}.prototxt".format(y))
        model = os.path.join(SphConv_ROOT,
                             "caffemodel",
                             "SphereH{0}Ks{1}".format(self.sphereH, self.ks),
                             "faster-rcnn{}".format(self.layer),
                             "tilt{0:03d}_iter_{1}.caffemodel".format(y, itr_fin))
        if not os.path.isfile(proto):
            raise ValueError("{} does not exist.".format(proto))
        if not os.path.isfile(model):
            raise ValueError("{} does not exist.".format(model))
        if silence:
            stderr = os.dup(2)
            devnull = os.open('/dev/null', os.O_WRONLY)
            os.dup2(devnull, 2)
        net = caffe.Net(proto, model, caffe.TEST)
        if silence:
            os.dup2(stderr, 2)
            os.close(devnull)
        return net

