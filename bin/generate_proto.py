#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
from caffe import layers as L, to_proto

from util.rf import rounded_rf, top_down
from util.network import use_BatchNorm

def Data(crop_size, data_path, batch_size=256, phase=0):
    crop_w, crop_h = crop_size
    param_str="{{\'im_shape\': [{0}, {1}], \'data_path\': \'{2}\', \'batch_size\': {3}}}".format(crop_h, crop_w, data_path, batch_size)
    data, label = L.Python(name="data", ntop=2,
                           python_param=dict(module="SphericalLayers",
                                             layer="SphericalIntermediateLayer",
                                             param_str=param_str),
                           include=dict(phase=phase))
    data.set_name("data")
    label.set_name("label")
    return data, label

def BatchNormScale(bottom, name_prefix):
    name = "{}/bn".format(name_prefix)
    kwargs = {
        "name": name,
        "moving_average_fraction": 0.9,
        "param": [dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0)],

    }
    if bottom is None:
        kwargs["bottom"] = "data"
        batchnorm = L.BatchNorm(**kwargs)
    else:
        batchnorm = L.BatchNorm(bottom, **kwargs)
    batchnorm.set_name(name)

    name = "{}/scale".format(name_prefix)
    scale = L.Scale(batchnorm, name=name,
                    bias_term=True,
                    param=[dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)])
    scale.set_name(name)
    return scale

def convolution(bottom, name_prefix, nout, ks=3, initialize=True):
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

    if initialize:
        kwargs["weight_filler"] = dict(type="gaussian", std=0.01)
        kwargs["bias_filler"] = dict(type="constant", value=0)
        kwargs["param"] = [dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)]

    if bottom is None:
        kwargs["bottom"] = "data"
        conv = L.Convolution(**kwargs)
    else:
        conv = L.Convolution(bottom, **kwargs)
    conv.set_name(name_prefix)
    return conv

def fullyconnected(bottom, name_prefix, nout=4096, initialize=True):
    kwargs = {
        "name": name_prefix,
        "num_output": nout,
    }
    if initialize:
        kwargs["weight_filler"] = dict(type="gaussian", std=0.01)
        kwargs["bias_filler"] = dict(type="constant", value=0)
        kwargs["param"] = [dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)]
    if bottom is None:
        kwargs["bottom"] = "data"
        fc = L.InnerProduct(**kwargs)
    else:
        fc = L.InnerProduct(bottom, **kwargs)
    fc.set_name(name_prefix)
    return fc

def Loss(bottom, label):
    loss = L.Python(bottom, label, name="loss",
                    module="SphericalLayers",
                    layer="EuclideanLossLayer",
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
        nout = 4096
    return nout

def layer_structure(data, layer, tilt, crop_out, initialize=True):
    if use_BatchNorm(layer):
        data = BatchNormScale(data, name_prefix="conv{0}/{1}".format(layer, tilt))
    if "_" in layer:
        nout = num_channels(layer)
        name_prefix = "conv{0}/{1}".format(layer, tilt)
        param = convolution(data, name_prefix, nout, ks=crop_out, initialize=initialize)
    else:
        name_prefix = "fc{}".format(layer)
        param = fullyconnected(data, name_prefix, initialize=initialize)
    return param

def deploy_layer(sphereH, layer, ks, tilt):
    # prepare input path / input shape
    _, crop_out = rounded_rf(layer, tilt, sphereH=sphereH, ks=ks)
    crop_w, crop_h = crop_out

    bot = top_down[layer]
    Nch = num_channels(bot)
    template = (
            "input: \"data\"\n"
            "input_shape {{\n"
            "  dim: 1\n"
            "  dim: {0}\n"
            "  dim: {1}\n"
            "  dim: {2}\n"
            "}}\n"
            )
    proto_str = template.format(Nch, crop_h, crop_w)

    # Data BatchNorm
    data = None
    param = layer_structure(data, layer, tilt, crop_out, initialize=False)
    proto_str += str(to_proto(param))
    return proto_str

def single_layer(sphereH, ks, network, layer, tilt):
    # prepare input path / input shape
    target_dir = "TargetSphereH{0}Ks{1}/{2}{3}".format(sphereH, ks, network, layer)
    template = os.path.join(cfg.DATA_ROOT, target_dir,
                            "tilt{:03d}.{{}}.h5".format(tilt))
    _, crop_out = rounded_rf(layer, tilt, sphereH=sphereH, ks=ks)

    # Datalayer train
    data_path = template.format("train")
    data, label = Data(crop_out, data_path, batch_size=256)
    proto_str = str(to_proto(data))

    # Datalayer test
    data_path = template.format("test")
    data, label = Data(crop_out, data_path, batch_size=64, phase=1)
    param = layer_structure(data, layer, tilt, crop_out)

    # Loss
    loss = Loss(param, label)
    proto_str += str(to_proto(loss))
    return proto_str

step = 1

def parse_solver(sphereH, ks, network, layer, tilt=None):
    proto_root = os.path.join(SphConv_ROOT, "prototxt")
    proto_dir = os.path.join(proto_root,
                             "SphereH{0}Ks{1}".format(sphereH, ks),
                             "{0}{1}".format(network, layer))
    if tilt is None and not os.path.isdir(proto_dir):
        os.makedirs(proto_dir)

    with open(os.path.join(proto_root, 'solver.prototxt'), 'r') as fin:
        solvers = fin.readlines()

    if tilt is None:
        tilts = range(0, sphereH, step)
    else:
        tilts = [tilt]

    caffemodel_dir = os.path.join(SphConv_ROOT, "caffemodel",
                                  "SphereH{0}Ks{1}".format(sphereH, ks),
                                  "{0}{1}".format(network, layer))
    if tilt is None and not os.path.isdir(caffemodel_dir):
        os.makedirs(caffemodel_dir)

    for t in tilts:
        path = os.path.join(proto_dir, "solver.tilt{:03d}.prototxt".format(t))
        if tilt is None and os.path.exists(path):
            sys.stderr.write("Skip {}\n".format(path))
            continue

        solver_str = ""
        for line in solvers:
            if line[0] == "#":
                continue
            if line.startswith("net:"):
                line = "net: \"{0}/net.tilt{1:03d}.prototxt\"\n".format(proto_dir, t)
            elif line.startswith("snapshot_prefix"):
                line = "snapshot_prefix: \"{0}/tilt{1:03d}\"\n".format(caffemodel_dir, t)
            solver_str += line

        if tilt is None:
            with open(path, 'w') as fout:
                fout.write(solver_str)
    return solver_str

def generate_proto(sphereH, ks, network, layer):
    parse_solver(sphereH, ks, network, layer)

    proto_dir = os.path.join(SphConv_ROOT, "prototxt",
                             "SphereH{0}Ks{1}".format(sphereH, ks),
                             "{0}{1}".format(network, layer))
    for tilt in xrange(0, sphereH, step):
        path = os.path.join(proto_dir, "net.tilt{:03d}.prototxt".format(tilt))
        if os.path.isfile(path):
            sys.stderr.write("Skip {}\n".format(path))
            continue
        proto_str = single_layer(sphereH=sphereH,
                                 ks=ks,
                                 network=network,
                                 layer=layer,
                                 tilt=tilt)
        with open(path, "w") as fout:
            fout.write(proto_str)

        path = os.path.join(proto_dir, "deploy.tilt{:03d}.prototxt".format(tilt))
        if os.path.isfile(path):
            sys.stderr.write("Skip {}\n".format(path))
            continue
        proto_str = deploy_layer(sphereH=sphereH,
                                 ks=ks,
                                 layer=layer,
                                 tilt=tilt)
        with open(path, "w") as fout:
            fout.write(proto_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', dest='prototxt',
                        choices=['net', 'deploy', 'solver', 'all'], default='deploy', type=str)
    parser.add_argument('--network', dest='network',
                        choices=['vgg16', 'faster-rcnn'], default='faster-rcnn', type=str)
    parser.add_argument('--sphereH', dest='sphereH', type=int, default=320)
    parser.add_argument('--ks', dest='ks', type=int, default=640)
    parser.add_argument('layer', choices=top_down.keys(), type=str)
    parser.add_argument('tilt', type=int)
    args = parser.parse_args()

    if args.prototxt == "net":
        proto_str = single_layer(sphereH=args.sphereH,
                                 ks=args.ks,
                                 network=args.network,
                                 layer=args.layer,
                                 tilt=args.tilt)
        print proto_str
    elif args.prototxt == "deploy":
        proto_str = deploy_layer(sphereH=args.sphereH,
                                 ks=args.ks,
                                 layer=args.layer,
                                 tilt=args.tilt)
        print proto_str
    elif args.prototxt == "solver":
        proto_str = parse_solver(sphereH=args.sphereH,
                                 ks=args.ks,
                                 network=args.network,
                                 layer=args.layer,
                                 tilt=args.tilt)
        print proto_str
    else:
        generate_proto(sphereH=args.sphereH,
                       ks=args.ks,
                       network=args.network,
                       layer=args.layer)

if __name__ == '__main__':
    main()
