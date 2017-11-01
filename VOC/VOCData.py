#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import xml.etree.ElementTree as ElementTree

from collections import namedtuple

VOCObject = namedtuple("VOCObject", ["name", "difficult", "xmin", "xmax", "ymin", "ymax"])

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

class VOCImage(object):
    def __init__(self, xml_path, voc_dir=None):
        root = ElementTree.parse(xml_path).getroot()

        name = root.findtext("filename")
        self.name = name
        self.path = os.path.join(voc_dir, "JPEGImages", name)

        sizes = root.findall("size")
        assert len(sizes) == 1
        W = int(sizes[0].findtext("width"))
        H = int(sizes[0].findtext("height"))
        C = int(sizes[0].findtext("depth"))
        self.shape = np.array([H, W, C])

        objs = []
        for obj in root.findall("object"):
            vocobj = parse_objannotation(obj)
            if vocobj.difficult:
                continue
            objs.append(vocobj)
        self.objs = objs

def parse_objannotation(obj):
    assert isinstance(obj, ElementTree.Element)
    name = obj.findtext("name")
    difficult = int(obj.findtext("difficult"))

    bboxes = obj.findall("bndbox")
    assert len(bboxes) == 1
    bbox = bboxes[0]
    xmin = int(bbox.findtext("xmin"))
    xmax = int(bbox.findtext("xmax"))
    ymin = int(bbox.findtext("ymin"))
    ymax = int(bbox.findtext("ymax"))
    vocobj = VOCObject(name, difficult, xmin, xmax, ymin, ymax)
    return vocobj

def load_VOCannotation(category="person", voc=2007):
    voc_root = "SphericalConvolution/ObjectDetection/VOCdevkit"
    voc_dir = os.path.join(voc_root, "VOC{}".format(voc))

    imageset_dir = os.path.join(voc_dir, "ImageSets", "Main")
    if voc == 2007:
        imageset = "test"
    else:
        imageset = "trainval"
    imageset_path = os.path.join(imageset_dir, "{0}_{1}.txt".format(category, imageset))

    xml_dir = os.path.join(voc_dir, "Annotations")
    imgs = []
    with open(imageset_path, 'r') as fin:
        for line in fin:
            imgId, label = line.rstrip().split()
            xml_path = os.path.join(xml_dir, "{}.xml".format(imgId))
            img = VOCImage(xml_path, voc_dir=voc_dir)
            if img.shape[2] != 3:
                sys.stdout.write("Only consider color image, ignore image {}\n".format(imgId))
                continue
            if len(img.objs) == 0:
                sys.stdout.write("Ignore difficult object, skip image {}\n".format(imgId))
                continue
            imgs.append(img)
    return imgs

VOCBBox = namedtuple("VOCBBox", ["frame", "label", "name", "xmin", "xmax", "ymin", "ymax"])

def voc_bboxes(voc=2007):
    imgs = load_VOCannotation(voc=voc)

    bboxes = []
    for img in imgs:
        frame = img.path
        frameId = os.path.splitext(img.name)[0]
        for i, obj in enumerate(img.objs):
            label = obj.name
            name = "{0}-bbox{1:03d}.jpg".format(frameId, i)
            bbox = VOCBBox(frame=frame, label=label, name=name,
                           xmin=obj.xmin, xmax=obj.xmax,
                           ymin=obj.ymin, ymax=obj.ymax)
            bboxes.append(bbox)
    return bboxes

VOC_ROOT = "SphericalConvolution/ObjectDetection/Pano"
IMG_SPHEREH = 640
IMG_KS = 640

def load_bboxes(tilt, voc, **kwargs):
    fov = kwargs.get("view_angle", 65.5)
    rf_size = kwargs.get("rf_size", 224)

    bbox_dir = os.path.join(VOC_ROOT,
                            "SphereH{0}FOV{1}Ks{2}Rf{3}".format(IMG_SPHEREH, int(fov), IMG_KS, rf_size),
                            "VOC{}".format(voc),
                            "tilt{:03d}".format(tilt))

    paths = []
    labels = []
    bboxes = []
    for bbox in voc_bboxes(voc=voc):
        img_path = os.path.join(bbox_dir, bbox.name)
        if os.path.isfile(img_path):
            cls_ind = CLASSES.index(bbox.label)
            labels.append(cls_ind)
            paths.append(img_path)
            bboxes.append(bbox)
    labels = np.array(labels, dtype=np.int32)
    return paths, labels, bboxes

if __name__ == "__main__":
    bboxes = voc_bboxes(voc=2007)
    for bbox in bboxes:
        print bbox.frame
        print bbox.label
        print bbox.name
        print "==="
    print len(bboxes)

