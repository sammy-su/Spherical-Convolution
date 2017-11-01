#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import numpy as np

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from cfg import DATA_ROOT
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from Extractor.SphConv import SphConv
from Extractor.SphericalConvolution import SphericalConvolution
from Extractor.ExactProjection import ExactProjection
from SphereProjection import SphereCoordinates
from util.network import load_network, forward_patch
from VOC.VOCData import load_bboxes, IMG_SPHEREH, CLASSES

def backproject_convolution(imgs, tilt, method, **kwargs):
    def evaluate_fc6(path):
        frameId = os.path.splitext(os.path.basename(path))[0]
        print frameId

        img = cv2.imread(path)
        if sphereH != img.shape[0]:
            img = cv2.resize(img, (sphereW, sphereH))
        data = np.zeros((kh, kw, Nch))
        xs = []
        ys = []
        idx = []
        for i in xrange(kh):
            for j in xrange(kw):
                x = Px[i,j]
                y = Py[i,j]
                x = int(round(x))
                y = int(round(y))
                xs.append(x)
                ys.append(y)
                idx.append((i, j))

        if method == "sphconv":
            base_path = os.path.join(base_dir, "{}.h5".format(frameId))
            vals = extractor.extract_convs(img, xs, ys, conv_path=base_path)
        else:
            vals = extractor.extract_convs(img, xs, ys)
        for (i, j), val in zip(idx, vals):
            data[i,j,:] = val
        data = np.transpose(data, (2,0,1))
        fc_net.blobs['data'].data[0] = data
        target = 'fc6'
        out = fc_net.forward(blobs=[target])
        fc6 = out[target].copy()
        return fc6

    sphereH = kwargs.get("sphereH", 640)
    sphereW = sphereH * 2
    fov = kwargs.get("view_angle", 65.5)
    ks = kwargs.get("ks", 640)

    rf = SphereCoordinates(kernel_size=224, sphereW=sphereW, sphereH=sphereH,
                           view_angle=fov, imgW=ks)
    Px, Py = rf.generate_grid(tilt)
    Px = Px[8::16,8::16]
    Py = Py[8::16,8::16]

    bot = "5_3"
    if method == "exact":
        extractor = ExactProjection(layer=bot, sphereH=sphereH, ks=ks, fov=fov)
    elif method == "optconv":
        extractor = SphericalConvolution(layer=bot, sphereH=sphereH, ks=ks, fov=fov)
    elif method == "sphconv":
        extractor = SphConv(layer=bot, sphereH=sphereH, ks=ks, fov=fov)
        root = os.path.join(DATA_ROOT, "PanoTarget")
        rf_size = kwargs["rf_size"]
        voc = kwargs["voc"]
        base_dir = os.path.join(root,
                                "Rf{}".format(rf_size),
                                "VOC{}".format(voc),
                                "tilt{:03d}".format(tilt))
    else:
        raise ValueError("Not implement")

    fc_net = load_network(layer="FC6", network='faster-rcnn')
    _, Nch, kh, kw = fc_net.blobs['data'].shape

    fc6 = np.vstack([evaluate_fc6(path) for path in imgs]).transpose()
    return fc6

def predict_scores(feature):
    def forward(W, b, v):
        vp = v.copy()
        vp[v<0.] = 0.
        out = np.dot(W, vp)
        out += b[:,None]
        return out

    net = load_network(layer="8", network="faster-rcnn")
    W_fc7 = net.params['fc7'][0].data
    b_fc7 = net.params['fc7'][1].data
    W_cls = net.params['cls_score'][0].data
    b_cls = net.params['cls_score'][1].data

    fc7 = forward(W_fc7, b_fc7, feature)
    score = forward(W_cls, b_cls, fc7)
    return score

def AP(scores, labels):
    aps = []
    for cls_ind, label in enumerate(CLASSES[1:], 1):
        cls_label = (labels == cls_ind).astype(np.int32)
        cls_score = scores[cls_ind,:]
        ap = average_precision_score(cls_label, cls_score)
        aps.append(ap)
        sys.stdout.write("{0} {1}\n".format(label, ap))
    ap = np.array(aps).mean()
    return ap

def ACC(scores, labels):
    f1scores = []

    # Ignore background
    pred_labels = np.argmax(scores[1:], axis=0) + 1
    acc = accuracy_score(labels, pred_labels)

    for cls_ind, label in enumerate(CLASSES[1:], 1):
        cls_label = (pred_labels == cls_ind).astype(np.int32)
        true_label = (labels == cls_ind).astype(np.int32)
        f1score = f1_score(true_label, cls_label)
        f1scores.append(f1score)
        sys.stdout.write("{0} {1}\n".format(label, f1score))
    f1score = np.array(f1scores).mean()
    return acc, f1score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest="method", type=str,
                        choices=['optconv', 'sphconv', 'exact'], default='exact')
    parser.add_argument('--stride', dest="stride", type=int, default=16)
    parser.add_argument('--scale', dest="scale", type=int, default=1)
    parser.add_argument('--rf', type=int, default=224)
    parser.add_argument('--sphereH', type=int, default=320)
    parser.add_argument('--ks', type=int, default=640)
    parser.add_argument('--persw', dest="persw", type=int, default=960)
    parser.add_argument('voc', type=int, choices=[2007, 2012], default=2007)
    parser.add_argument('tilt', type=int, default=320)
    args = parser.parse_args()

    kwargs = {
        "sphereH": args.sphereH,
        "view_angle": 65.5,
        "ks": args.ks,
        "rf_size": args.rf,
        "stride": args.stride,
        "scale": args.scale,
        "persw": args.persw,
    }
    orig_tilt = args.tilt * IMG_SPHEREH / args.sphereH
    paths, labels, _ = load_bboxes(orig_tilt, args.voc, **kwargs)
    if len(paths) == 0:
        sys.stderr.write("No projected image found.\n")
        return

    kwargs["voc"] = args.voc
    fc6 = backproject_convolution(paths, args.tilt, args.method, **kwargs)

    scores = predict_scores(fc6)
    ap = AP(scores, labels)
    acc, f1score = ACC(scores, labels)
    sys.stdout.write("{0:.4f} {1:.4f} {2:.4f}\n".format(ap, acc, f1score))

if __name__ == "__main__":
    main()

