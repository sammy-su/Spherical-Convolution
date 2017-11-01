#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import multiprocessing
import numpy as np

from functools import partial

kernel_sizes = {
    '1_1': 3,
    '1_2': 5,
    '2_1': 10,
    '2_2': 14,
    '3_1': 24,
    '3_2': 32,
    '3_3': 40,
    '4_1': 60,
    '4_2': 76,
    '4_3': 92,
    '5_1': 132,
    '5_2': 164,
    '5_3': 196,
    '6': 224,
}

top_down ={
    '1_1': 'pixel',
    '1_2': '1_1',
    '2_1': '1_2',
    '2_2': '2_1',
    '3_1': '2_2',
    '3_2': '3_1',
    '3_3': '3_2',
    '4_1': '3_3',
    '4_2': '4_1',
    '4_3': '4_2',
    '5_1': '4_3',
    '5_2': '5_1',
    '5_3': '5_2',
    '6': '5_3',
}

strides = {
    640: {
        640: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 1,
            '2_2': 1,
            '3_1': 1,
            '3_2': 1,
            '3_3': 1,
            '4_1': 2,
            '4_2': 2,
            '4_3': 2,
            '5_1': 4,
            '5_2': 4,
            '5_3': 4,
            '6': 4,
        },
        320: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 1,
            '2_2': 1,
            '3_1': 1,
            '3_2': 1,
            '3_3': 1,
            '4_1': 1,
            '4_2': 1,
            '4_3': 1,
            '5_1': 2,
            '5_2': 2,
            '5_3': 2,
            '6': 2,
        },
        160: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 1,
            '2_2': 1,
            '3_1': 1,
            '3_2': 1,
            '3_3': 1,
            '4_1': 1,
            '4_2': 1,
            '4_3': 1,
            '5_1': 1,
            '5_2': 1,
            '5_3': 1,
            '6': 1,
        },
    },
    224: {
        640: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 2,
            '2_2': 2,
            '3_1': 4,
            '3_2': 4,
            '3_3': 4,
            '4_1': 8,
            '4_2': 8,
            '4_3': 8,
            '5_1': 16,
            '5_2': 16,
            '5_3': 16,
            '6': 16,
        },
        320: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 1,
            '2_2': 1,
            '3_1': 2,
            '3_2': 2,
            '3_3': 2,
            '4_1': 4,
            '4_2': 4,
            '4_3': 4,
            '5_1': 8,
            '5_2': 8,
            '5_3': 8,
            '6': 8,
        },
        160: {
            '1_1': 1,
            '1_2': 1,
            '2_1': 1,
            '2_2': 1,
            '3_1': 1,
            '3_2': 1,
            '3_3': 1,
            '4_1': 2,
            '4_2': 2,
            '4_3': 2,
            '5_1': 4,
            '5_2': 4,
            '5_3': 4,
            '6': 4,
        }
    },
}

class ReceptiveField(object):
    def __init__(self, kernel_size=3, sphereW=1280, sphereH=640, view_angle=65.5, imgW=640):
        self.sphereW = sphereW
        self.sphereH = sphereH
        self.kernel_size = kernel_size

        TX, TY = self._meshgrid()
        kernel_angle = kernel_size * view_angle / imgW
        R, ANGy = self._compute_radius(kernel_angle, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        if self.kernel_size % 2 == 1:
            center = self.kernel_size / 2
        else:
            center = self.kernel_size / 2 - 0.5
        TX, TY = np.meshgrid(range(self.kernel_size), range(self.kernel_size))
        TX = TX.astype(np.float64) - center
        TY = TY.astype(np.float64) - center
        return TX, TY

    def _compute_radius(self, angle, TY):
        _angle = np.pi * angle / 180.
        r = 0.5 * self.kernel_size / np.tan(_angle/2.)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def generate_grid(self, tilt):
        if not self.sphereH > tilt >= 0:
            raise ValueError("Invalid polar displace")
        rotate_y = (self.sphereH / 2. - 0.5 - tilt) * np.pi / self.sphereH
        Px, Py = self._sample_points(rotate_y)
        return Px, Py

    def _sample_points(self, rotate_y):
        angle_x, angle_y = self._direct_camera(rotate_y)
        Px = angle_x / (2*np.pi) * self.sphereW
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH - 0.5
        return Px, Py

    def _direct_camera(self, rotate_y):
        angle_y = self._ANGy + rotate_y
        # Padding great circle
        INDn = np.abs(angle_y) > np.pi/2

        X = np.sin(angle_y) * self._R
        Y = - np.cos(angle_y) * self._R
        Z = self._Z
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)

        angle_x = np.arctan(Z / -Y)
        angle_y = np.arctan(X / RZY)

        # padding
        angle_x[INDn] += np.pi
        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y

rf_path = os.path.join(os.path.abspath(os.path.curdir), "rf.yaml")

def rfshape(tilt, layer, sphereH=640, ks=640):
    if not hasattr(rfshape, "shapes"):
        if os.path.isfile(rf_path):
            with open(rf_path, 'r') as fin:
                shapes = yaml.load(fin)
            rfshape.shapes = shapes
        else:
            raise ValueError("{} does not exist".format(rf_path))
    if not sphereH > tilt >= 0:
        raise ValueError("Invalid tilt.")
    if tilt >= sphereH/2:
        tilt = sphereH - tilt - 1
    key = "SphereH{0}Ks{1}".format(sphereH, ks)
    shape = rfshape.shapes[key][layer][tilt]
    return shape

def rounded_rf(layer, tilt, sphereH=640, ks=640):
    crop_w, crop_h = rfshape(tilt, layer, sphereH=sphereH, ks=ks)
    stride = strides[ks][sphereH][layer]

    d_stride = 2 * stride
    ns_w = int(np.ceil((crop_w-1.) / d_stride))
    ns_h = int(np.ceil((crop_h-1.) / d_stride))

    crop_in = (ns_w * d_stride + 1, ns_h * d_stride + 1)
    crop_out = (ns_w * 2 + 1, ns_h * 2 + 1)
    return crop_in, crop_out

def compute_covers(tilt, layer, top_rf, sphereH=640, ks=640):
    def covered_pixels(Px, Py):
        pixels = set()
        for i in xrange(Px.shape[0]):
            for j in xrange(Px.shape[1]):
                x = Px[i,j]
                y = Py[i,j]
                xu = int(np.ceil(x))
                xd = int(np.floor(x))
                yu = int(np.ceil(y))
                yd = int(np.floor(y))
                pixels.add((xu, yu))
                pixels.add((xu, yd))
                pixels.add((xd, yu))
                pixels.add((xd, yd))
        return pixels

    def translate_coord(i, j):
        if j < 0:
            Pi = i + sphereW / 2
            Pj = -1 - j
        elif j >= sphereH:
            Pi = i + sphereW / 2
            Pj = 2 * sphereH - j - 1
        else:
            Pj = j
            Pi = i
        if Pi >= sphereW:
            Pi -= sphereW
        elif Pi < 0:
            Pi += sphereW
        return Pi, Pj

    def bottom_coverage(Ci, Cj, W, H):
        bottom = np.zeros((sphereH, sphereW), dtype=np.int32)
        j_start = Cj - H/2
        j_end = j_start + H
        i_start = Ci - W/2
        i_end = i_start + W
        for j in xrange(j_start, j_end):
            for i in xrange(i_start, i_end):
                Pi, Pj = translate_coord(i, j)
                bottom[Pj,Pi] = 1
        return bottom

    if layer not in top_down:
        raise ValueError("Invalid layer.")
    sphereW = sphereH * 2
    center = sphereW / 2

    # Compute the pixels covered by target Rf
    kernel_size = kernel_sizes[layer]
    target_rf = ReceptiveField(kernel_size=kernel_size,
                               sphereW=sphereW,
                               sphereH=sphereH,
                               imgW=ks)
    Px, Py = target_rf.generate_grid(tilt)
    target_pixels = covered_pixels(Px+center, Py)
    target_covered = np.zeros((sphereH, sphereW), dtype=np.int32)
    for x, y in target_pixels:
        i, j = translate_coord(x, y)
        target_covered[j,i] = 1

    rf = ReceptiveField(kernel_size=3,
                        sphereW=sphereW,
                        sphereH=sphereH,
                        imgW=ks)
    W, H = top_rf
    top = top_down[layer]
    bottom_covered = bottom_coverage(center, tilt, W, H)
    while True:
        bottom = top_down[top]
        Ij, Ii = bottom_covered.nonzero()
        bottom_covered = np.zeros((sphereH, sphereW), dtype=np.int32)
        if bottom == 'pixel':
            grids = {}
            for Cj, Ci in zip(Ij, Ii):
                if Cj in grids:
                    Px, Py = grids[Cj]
                else:
                    Px, Py = rf.generate_grid(Cj)
                    grids[Cj] = (Px, Py)
                pixels = covered_pixels(Px+Ci, Py)
                for x, y in pixels:
                    i, j = translate_coord(x, y)
                    bottom_covered[j,i] = 1
            result_covered = bottom_covered
            break
        else:
            for Cj, Ci in zip(Ij, Ii):
                W, H = rfshape(Cj, top, sphereH=sphereH, ks=ks)
                # bottle neck in the following two line
                bottom_covered_ = bottom_coverage(Ci, Cj, W, H)
                bottom_covered = np.logical_or(bottom_covered, bottom_covered_)
            top = bottom

    covered = np.logical_and(target_covered, result_covered)
    not_covered = target_covered - covered
    not_covered_count = not_covered.sum()
    return covered, not_covered, result_covered

def build_tiltRf(tilt, layer, sphereH, ks=640):
    stride = strides[ks][sphereH][layer]
    W = 2 * stride + 1
    H = 2 * stride + 1
    while True:
        top_rf = (W, H)
        nw = (W - 1) / stride + 1
        nh = (H - 1) / stride + 1
        covered, not_covered, result_covered = compute_covers(tilt,
                                                              layer,
                                                              top_rf,
                                                              sphereH=sphereH,
                                                              ks=ks)
        covered_count = covered.sum()
        coverage = float(covered_count) / (covered_count + not_covered.sum())
        threshold = 0.95
        if coverage > threshold:
            break

        if nw * nh > 7*7:
            sys.stdout.write("Reach kernel size upper bound.\n")
            break

        Cj, Ci = result_covered.nonzero()
        NCj, NCi = not_covered.nonzero()
        if NCj.max() > Cj.max() or NCj.min() < Cj.min():
            H += 2 * stride
            continue
        if NCi.max() > Ci.max() or NCi.min() < Ci.min():
            W += 2 * stride
            continue
        sys.stdout.write("Ignore internal holes.\n")
        break
    print tilt, top_rf, "-> ({0}, {1})".format(nw, nh)
    return tilt, top_rf

def build_rf(layer, sphereH=640, ks=640):
    if os.path.isfile(rf_path):
        with open(rf_path, 'r') as fin:
            shapes = yaml.load(fin)
    else:
        shapes = {}

    key = "SphereH{0}Ks{1}".format(sphereH, ks)
    Hshapes = shapes.setdefault(key, dict())
    if layer in Hshapes:
        sys.stderr.write("Layer {} exists.\n".format(layer))
        return

    bot = top_down[layer]
    if bot not in Hshapes and bot != "1_1":
        sys.stderr.write("Build layer {} first.\n".format(bot))
        return

    pool = multiprocessing.Pool()
    build_tiltRf_ = partial(build_tiltRf, layer=layer, sphereH=sphereH, ks=ks)
    results = pool.map(build_tiltRf_, range(sphereH/2,-1,-1))
    pool.close()
    pool.join()

    tilt_shapes = {}
    for tilt, top_rf in results:
        tilt_shapes[tilt] = top_rf
    Hshapes[layer] = tilt_shapes

    with open(rf_path, 'w') as fout:
        yaml.dump(shapes, fout)
    return

if __name__ == "__main__":
    layers = ["1_2", "2_1", "2_2", "3_1", "3_2", "3_3", "4_1", "4_2", "4_3", "5_1", "5_2", "5_3", "6"]
    for layer in layers:
        build_rf(layer, sphereH=160, ks=640)

