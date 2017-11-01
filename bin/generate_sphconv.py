#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg

from cfg import DATA_ROOT
from caffe import layers as L, to_proto
from util.rf import rounded_rf, top_down, strides
from util.network import use_BatchNorm

def Data(layer, phase=0, sphereH=320, ks=640, network='faster-rcnn'):
    param_template = (
        "{{\'layer\': \'{0}\',"
        "\'sphereH\': {1},"
        "\'ks\': {2},"
        "\'network\': \'{3}\',"
        "\'phase\': \'{4}\',"
        "\'prefetch\': {5}}}"
    )
    if not phase:
        phase_str = "train"
        prefetch = 1
    else:
        phase_str = "test"
        prefetch = 1
    param_str = param_template.format(layer, sphereH, ks, network, phase_str, prefetch)
    data, label = L.Python(name="data", ntop=2,
                           python_param=dict(module="SphericalLayers",
                                             layer="SphConvDataLayer",
                                             param_str=param_str),
                           include=dict(phase=phase))
    data.set_name("data")
    label.set_name("label")
    return data, label

def BatchNormScale(bottom, name_prefix, deploy=True):
    name = "{}/bn".format(name_prefix)
    kwargs = {
        "name": name,
        "moving_average_fraction": 0.9,
        "param": [dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0)],
        "use_global_stats": True,
    }
    if bottom is None:
        kwargs["bottom"] = "data"
        batchnorm = L.BatchNorm(**kwargs)
    else:
        batchnorm = L.BatchNorm(bottom, **kwargs)
    batchnorm.set_name(name)

    name = "{}/scale".format(name_prefix)
    if deploy:
        scale = L.Scale(batchnorm, name=name,
                        bias_term=True,
                        param=[dict(lr_mult=0, decay_mult=0),
                               dict(lr_mult=0, decay_mult=0)])
    else:
        scale = L.Scale(batchnorm, name=name,
                        bias_term=True,
                        param=[dict(lr_mult=0, decay_mult=0),
                               dict(lr_mult=0, decay_mult=0)])
    scale.set_name(name)
    return scale

def Convolution(bottom, name_prefix, nout, ks=3, deploy=True):
    kwargs = {
        "name": name_prefix,
        "num_output": nout,
    }
    if isinstance(ks, int):
        kwargs["kernel_size"] = ks
    elif isinstance(ks, tuple):
        kw, kh = ks
        kwargs["kernel_w"] = kw
        kwargs["kernel_h"] = kh
    if not deploy:
        kwargs["param"] = [dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)]
    else:
        kwargs["param"] = [dict(lr_mult=0, decay_mult=0),
                           dict(lr_mult=0, decay_mult=0)]
    if bottom is None:
        kwargs["bottom"] = "data"
        conv = L.Convolution(**kwargs)
    else:
        conv = L.Convolution(bottom, **kwargs)
    conv.set_name(name_prefix)
    return conv

def SplitRow(bottom, name_prefix, layer, sphereH=320, ks=640):
    stride = strides[ks][sphereH][layer]
    H = sphereH / stride
    param_str="{{\'layer\': \'{0}\', \'sphereH\': {1}, \'ks\': {2}}}".format(layer, sphereH, ks)
    python_param = dict(module="SphericalLayers",
                        layer="SplitRowLayer",
                        param_str=param_str)
    kwargs = {
        "name": "{}/split".format(name_prefix),
        "ntop": H,
        "python_param": python_param,
    }
    if bottom is None:
        kwargs["bottom"] = "data"
        rows = L.Python(**kwargs)
    else:
        rows = L.Python(bottom, **kwargs)
    for i, row in enumerate(rows):
        name = "{0}/r{1:03d}".format(name_prefix, i)
        row.set_name(name)
    return rows

def MergeRow(bottoms, name_prefix):
    python_param = dict(module="SphericalLayers",
                        layer="MergeRowLayer")
    kwargs = {
        "name": "{}/merge".format(name_prefix),
        "python_param": python_param,
    }
    plane = L.Python(*bottoms, **kwargs)
    plane.set_name("{}/merge".format(name_prefix))
    return plane

def ReLU(bottom, layer):
    if bottom is None:
        kwargs = {
            "name": "relu1_1",
            "in_place": False,
            "bottom": "data",
        }
        relu = L.ReLU(**kwargs)
        relu.set_name("data")
    else:
        bot = top_down[layer]
        kwargs = {
            "name": "relu{}".format(bot),
            "in_place": True,
        }
        relu = L.ReLU(bottom, **kwargs)
        relu.set_name(bottom.name)
    return relu

def Pool(bottom, layer):
    name = "pool{}".format(layer[:1])
    kwargs = {
        "pool": 0,
        "kernel_size": 2,
        "stride": 2,
        "name": name,
    }
    plane = L.Pooling(bottom, **kwargs)
    plane.set_name(name)
    return plane

def Loss(bottom, label):
    loss = L.Python(bottom, label, name="loss",
                    module="SphericalLayers",
                    layer="SphConvLossLayer",
                    loss_weight=1)
    loss.set_name("loss")
    return loss

def num_channels(layer):
    if "_" in layer:
        meta_layer = layer[:1]
        if meta_layer == "1":
            nout = 64
        elif meta_layer == "2":
            nout = 128
        elif meta_layer == "3":
            nout = 256
        else:
            nout = 512
    else:
        raise ValueError("Does not support FC layer.")
    return nout

def layer_structure(bottom, layer, sphereH=320, ks=640, deploy=True):
    nout = num_channels(layer)
    stride = strides[ks][sphereH][layer]

    prefix = "conv{0}".format(layer)
    relu = ReLU(bottom, layer)
    if layer == "5_1":
        bot = top_down[layer]
        relu = Pool(relu, bot)
    rows = SplitRow(relu, prefix, layer)
    convs = []
    for i, row in enumerate(rows):
        tilt = i * stride
        name_prefix = "conv{0}/{1}".format(layer, tilt)
        _, crop_out = rounded_rf(layer, tilt, sphereH=sphereH, ks=ks)
        if use_BatchNorm(layer):
            row = BatchNormScale(row, name_prefix, deploy=deploy)
            conv = Convolution(row, name_prefix, nout, ks=crop_out, deploy=deploy)
        else:
            conv = Convolution(row, name_prefix, nout, ks=crop_out, deploy=deploy)
        convs.append(conv)
    plane = MergeRow(convs, prefix)
    return plane

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', choices=['net', 'deploy'], default='deploy', type=str)
    parser.add_argument('--sphereH', type=int, default=320)
    parser.add_argument('--ks', type=int, default=640)
    parser.add_argument('layer', choices=top_down.keys(), type=str)
    args = parser.parse_args()

    layer = args.layer
    layers = []
    while layer != "1_1":
        layers.append(layer)
        layer = top_down[layer]
    layers.reverse()

    if args.prototxt == "deploy":
        template = (
            "input: \"data\"\n"
            "input_shape {{\n"
            "  dim: 1\n"
            "  dim: 64\n"
            "  dim: {0}\n"
            "  dim: {1}\n"
            "}}\n"
        )
        sphereW = args.sphereH * 2
        proto_str = template.format(args.sphereH, sphereW)
        plane = None
        deploy = True
    elif args.prototxt == "net":
        plane, label = Data(args.layer, phase=0, sphereH=args.sphereH, ks=args.ks)
        proto_str = str(to_proto(plane))
        plane, label = Data(args.layer, phase=1, sphereH=args.sphereH, ks=args.ks)
        deploy = False
    else:
        raise ValueError("Invalid prototxt.")

    for layer in layers:
        plane = layer_structure(plane, layer, sphereH=args.sphereH, ks=args.ks, deploy=deploy)

    if args.prototxt == "net":
        plane =  Loss(plane, label)
    proto_str += str(to_proto(plane))
    print proto_str

if __name__ == '__main__':
    main()
