
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

from Extractor.SphConv import SphConv
from Extractor.SphericalConvolution import SphericalConvolution
from Extractor.ExactProjection import ExactProjection
from SphereProjection import SphereCoordinates
from util.network import load_network, forward_patch
from util.faster_rcnn import generate_anchors, bbox_transform_inv, clip_boxes
from VOC.VOCData import load_bboxes, IMG_SPHEREH

def generate_proposals(imgs, tilt, method, **kwargs):
    def img_proposals(path):
        frameId = os.path.splitext(os.path.basename(path))[0]
        print frameId

        img = cv2.imread(path)
        if sphereH != img.shape[0]:
            img = cv2.resize(img, (sphereW, sphereH))

        data = np.zeros((Px.shape[0], Px.shape[1], Nch))
        xs = []
        ys = []
        idx = []
        for i in xrange(Px.shape[0]):
            for j in xrange(Px.shape[1]):
                x = Px[i,j]
                y = Py[i,j]
                x = int(round(x))
                y = int(round(y))
                xs.append(x)
                ys.append(y)
                idx.append((i, j))
        if method == "sphconv":
            frameId = os.path.splitext(os.path.basename(path))[0]
            base_path = os.path.join(base_dir, "{}.h5".format(frameId))
            vals = extractor.extract_convs(img, xs, ys, conv_path=base_path)
        else:
            vals = extractor.extract_convs(img, xs, ys)
        for (i, j), val in zip(idx, vals):
            data[i,j,:] = val
        data = np.transpose(data, (2,0,1))

        rpn_net.blobs['data'].data[0] = data
        targets = ['rpn_cls_prob_reshape', 'rpn_bbox_pred']
        out = rpn_net.forward(blobs=targets)
        scores = out['rpn_cls_prob_reshape']
        scores = scores[:,num_anchors:,:,:].reshape((num_anchors,1))
        bbox_deltas = out['rpn_bbox_pred']
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, ks=ks)
        proposals = np.hstack([proposals, scores])
        return proposals

    sphereH = kwargs.get("sphereH", 640)
    sphereW = sphereH * 2
    fov = kwargs.get("view_angle", 65.5)
    ks = kwargs.get("ks", 640)

    rf = SphereCoordinates(kernel_size=224, sphereW=sphereW, sphereH=sphereH,
                           view_angle=fov, imgW=ks)
    Px, Py = rf.generate_grid(tilt)
    Px = Px[8::16,8::16]
    Py = Py[8::16,8::16]
    Px = Px[5:8,5:8]
    Py = Py[5:8,5:8]
    anchor_scales = np.array((8, 16, 32))
    anchors = generate_anchors(scales=anchor_scales)
    anchors += ks/2
    num_anchors = anchors.shape[0]

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

    rpn_net = load_network(layer="RPN", network='faster-rcnn')
    Nch = rpn_net.blobs['data'].shape[1]
    proposals = {path: img_proposals(path) for path in imgs}
    return proposals

def bbox_IoU(proposals, bbox):
    # intersection
    ixmin = np.maximum(proposals[:, 0], bbox[0])
    iymin = np.maximum(proposals[:, 1], bbox[1])
    ixmax = np.minimum(proposals[:, 2], bbox[2])
    iymax = np.minimum(proposals[:, 3], bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
           (proposals[:, 2] - proposals[:, 0] + 1.) *
           (proposals[:, 3] - proposals[:, 1] + 1.) - inters)
    ious = inters / uni
    iou = np.max(ious)
    return iou

def scale_bbox(bbox, rf, ks):
    bbox_w = bbox.xmax - bbox.xmin
    bbox_h = bbox.ymax - bbox.ymin
    if bbox_w > bbox_h:
        scale = float(rf) / bbox_w
    else:
        scale = float(rf) / bbox_h

    W = int(np.floor(bbox_w * scale))
    H = int(np.floor(bbox_h * scale))
    center = ks / 2
    xmin = center - W/2
    xmax = xmin + W
    ymin = center - H/2
    ymax = ymin + H
    xmin = max(0, xmin)
    xmax = min(ks, xmax)
    ymin = max(0, ymin)
    ymax = min(ks, ymax)
    bbox = np.array([xmin, ymin, xmax, ymax])
    return bbox

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
    paths, _, bboxes = load_bboxes(orig_tilt, args.voc, **kwargs)
    if len(paths) == 0:
        sys.stderr.write("No projected image found.\n")
        return

    kwargs["voc"] = args.voc
    proposals = generate_proposals(paths, args.tilt, args.method, **kwargs)

    ious = []
    for i, img in enumerate(proposals):
        proposal = proposals[img]
        bbox = bboxes[i]
        bbox = scale_bbox(bbox, args.rf, args.ks)
        scores = proposal[:,-1]
        idx = scores > 0.5
        if idx.sum() > 0:
            proposal = proposal[idx,:4]
            iou = bbox_IoU(proposal, bbox)
        else:
            iou = 0.
        ious.append(iou)
    sys.stdout.write("{}\n".format(np.mean(np.array(ious))))

if __name__ == "__main__":
    main()
