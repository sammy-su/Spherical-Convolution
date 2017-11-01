
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
from SphereProjection import SphereProjection
from util.rf import kernel_sizes
from util.network import load_network, forward_patch
from util.data_io import sample_pixels, dump_pkl

def generate_targets(sphereH, network, layer, tilt, ks):
    # prepare output directory
    kernel_size = kernel_sizes[layer]
    if kernel_size > ks:
        sys.stder.write("Image size should be larger than receptive field.")
        sys.stder.write("Ks: {0} -> {1}\n".format(ks, kernel_size))
        ks = kernel_size

    target_dir = os.path.join(DATA_ROOT,
                              "TargetSphereH{0}Ks{1}".format(sphereH, ks),
                              "{0}{1}".format(network, layer))
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # Check if output exists
    train_path = os.path.join(target_dir, 'tilt{0:03d}.train.pkl'.format(tilt))
    build_train = not os.path.exists(train_path)
    test_path = os.path.join(target_dir, 'tilt{0:03d}.test.pkl'.format(tilt))
    build_test = not os.path.exists(test_path)
    if not build_train and not build_test:
        return

    # prepare projection matrix
    sphereW = sphereH * 2
    projection = SphereProjection(kernel_size=kernel_size,
                                  sphereW=sphereW,
                                  sphereH=sphereH,
                                  imgW=ks)
    P = projection.buildP(tilt)

    train = []
    test = []
    net = load_network(layer=layer, network=network, silence=False)
    samples = sample_pixels(tilt, sphereH=sphereH)
    for path, pixels in samples.viewitems():
        print os.path.basename(path)
        img = cv2.imread(path)
        if sphereH != img.shape[0]:
            img = cv2.resize(img, (sphereW, sphereH))
        if "mountain_climbing" in path:
            if not build_test:
                continue
            targets = test
        else:
            if not build_train:
                continue
            targets = train

        for x, y in pixels:
            assert y == tilt
            rimg = np.roll(img, sphereW/2-x, axis=1)
            patch = projection.project(P, rimg)
            val = forward_patch(patch, net, layer=layer)
            target = (path, x, y, val.ravel().tolist())
            targets.append(target)

    if build_train:
        dump_pkl(train, train_path)
    if build_test:
        dump_pkl(test, test_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sphereH', dest='sphereH', type=int, default=320)
    parser.add_argument('--ks', dest='ks', type=int, default=640)
    parser.add_argument('--network', dest='network',
                        choices=['vgg16', 'faster-rcnn'], default='faster-rcnn', type=str)
    parser.add_argument('layer', choices=kernel_sizes.keys(), type=str)
    parser.add_argument('tilt', type=int)
    args = parser.parse_args()

    if not args.sphereH > args.tilt >= 0:
        sys.stderr.write("Invalid tilt: {}\n".format(args.tilt))
        return
    generate_targets(args.sphereH, args.network, args.layer, args.tilt, args.ks)

if __name__ == "__main__":
    main()
