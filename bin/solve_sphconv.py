#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from util.rf import top_down
from util.network import init_sphconv

LOG_ROOT = os.path.join(SphConv_ROOT, "Log/solve_net")
if not os.path.isdir(LOG_ROOT):
    os.makedirs(LOG_ROOT)

def solve_sphconv(layer, network='faster-rcnn', log=False, random_init=False, state=None):
    log_dir = os.path.join(LOG_ROOT, "SphConv")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "{0}{1}.log".format(network, layer))
    if os.path.isfile(log_path):
        sys.stderr.write("Log already exists, please delete log file first.\n")
        return

    if log:
        stderr = os.dup(2)
        log = os.open(log_path, os.O_WRONLY | os.O_CREAT)
        os.dup2(log, 2)

    solver_proto = os.path.join(SphConv_ROOT, "prototxt", "SphConv",
                                "{0}{1}.solver.prototxt".format(network, layer))
    solver = caffe.get_solver(solver_proto)
    if state is not None:
        solver.restore(state)
    elif not random_init:
        init_sphconv(solver.net, layer, pretrained=False)
    solver.solve()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest="log", action="store_true")
    parser.add_argument('--rand', dest="random_init", action="store_true")
    parser.add_argument('--network', dest='network',
                        choices=['vgg16', 'faster-rcnn'], default='faster-rcnn', type=str)
    parser.add_argument('--state', dest='state', type=str, default=None)
    parser.add_argument('layer', choices=top_down.keys(), type=str)
    args = parser.parse_args()

    solve_sphconv(args.layer, network=args.network, log=args.log, random_init=args.random_init, state=args.state)

if __name__ == "__main__":
    main()
