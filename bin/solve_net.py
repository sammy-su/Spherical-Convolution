
import os
import sys
import argparse

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from cfg import DATA_ROOT
from util.rf import top_down

LOG_ROOT = os.path.join(SphConv_ROOT, "Log/solve_net")
if not os.path.isdir(LOG_ROOT):
    os.makedirs(LOG_ROOT)

def solve_net(args):
    src_dir = os.path.join(DATA_ROOT,
                           "TargetSphereH{0}Ks{1}".format(args.sphereH, args.ks),
                           "{0}{1}".format(args.network, args.layer))
    src_train = os.path.join(src_dir, "tilt{:03d}.train.h5".format(args.tilt))
    src_test = os.path.join(src_dir, "tilt{:03d}.test.h5".format(args.tilt))
    if not os.path.isfile(src_train):
        sys.stderr.write("Source {} does not exist.\n".format(src_train))
        return
    if not os.path.isfile(src_test):
        sys.stderr.write("Source {} does not exist.\n".format(src_test))
        return

    log_dir = os.path.join(LOG_ROOT, "SphereH{0}Ks{1}".format(args.sphereH, args.ks),
                           "{0}{1}".format(args.network, args.layer))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "tilt{0:03d}.log".format(args.tilt))

    model_root = os.path.join(DATA_ROOT, "Code", "caffemodel")
    itr_fin = 4000
    model_path = os.path.join(model_root,
                              "SphereH{0}Ks{1}".format(args.sphereH, args.ks),
                              "{0}{1}".format(args.network, args.layer),
                              "tilt{0:03d}_iter_{1}.caffemodel".format(args.tilt, itr_fin))
    if os.path.isfile(model_path):
        sys.stderr.write("Model {} exists.\n".format(model_path))
        return

    if os.path.isfile(log_path):
        sys.stderr.write("Log already exists, please delete log file first.\n")
        return
    stderr = os.dup(2)
    log = os.open(log_path, os.O_WRONLY | os.O_CREAT)
    os.dup2(log, 2)

    solver_proto = os.path.join(SphConv_ROOT, "prototxt",
                                "SphereH{0}Ks{1}".format(args.sphereH, args.ks),
                                "{0}{1}".format(args.network, args.layer),
                                "solver.tilt{:03d}.prototxt".format(args.tilt))
    solver = caffe.get_solver(solver_proto)
    solver.solve()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sphereH', dest='sphereH', type=int, default=320)
    parser.add_argument('--ks', dest='ks', type=int, default=640)
    parser.add_argument('--network', dest='network',
                        choices=['vgg16', 'faster-rcnn'], default='faster-rcnn', type=str)
    parser.add_argument('layer', choices=top_down.keys(), type=str)
    parser.add_argument('tilt', type=int)
    args = parser.parse_args()

    solve_net(args)

if __name__ == "__main__":
    main()
