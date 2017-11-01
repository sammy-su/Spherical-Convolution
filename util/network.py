
import os
import sys
import h5py
import numpy as np

SphConv_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SphConv_ROOT)

import cfg
import caffe

from util.rf import top_down, strides

def load_network(layer="1_1", network="faster-rcnn", silence=True):
    proto_root = os.path.join(SphConv_ROOT, "prototxt")
    model_proto = os.path.join(proto_root, 'layers', '{0}{1}.prototxt'.format(network, layer))
    if network == "vgg16":
        model_bin = os.path.join(cfg.caffe_root, "models/vgg16/VGG_ILSVRC_16_layers.caffemodel")
    elif network in ["faster-rcnn", "faster-rcnn-dilate", "faster-rcnn-pad"]:
        model_bin = "SphericalConvolution/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
    else:
        raise ValueError("Unknown network")

    if silence:
        stderr = os.dup(2)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, 2)
    net = caffe.Net(model_proto, model_bin, caffe.TEST)
    if silence:
        os.dup2(stderr, 2)
        os.close(devnull)
    return net

def forward_patch(patch, net, layer='1_1'):
    patch = patch - np.array([103.939, 116.779, 123.68])
    patch = np.transpose(patch, (2,0,1))
    if "_" in layer:
        target = 'conv{}'.format(layer)
    else:
        target = 'fc{}'.format(layer)
    net.blobs['data'].data[...] = patch
    out = net.forward(blobs=[target])
    return out[target].copy()


def load_sphconv(layer, sphereH=320, ks=640, network="faster-rcnn", silence=False, pretrained=True):
    proto_root = os.path.join(SphConv_ROOT, "prototxt", "SphConv")
    model_proto = os.path.join(proto_root, "{0}{1}.deploy.prototxt".format(network, layer))

    if silence:
        stderr = os.dup(2)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, 2)
    net = caffe.Net(model_proto, caffe.TEST)
    init_sphconv(net, layer, sphereH=sphereH, ks=ks, network=network, pretrained=pretrained)
    if silence:
        os.dup2(stderr, 2)
        os.close(devnull)
    return net

def use_BatchNorm(layer):
    if layer in ["5_3", "5_2", "5_1", "4_3", "4_2", "4_1"]:
        return True
    else:
        return False

def init_sphconv(net, layer, sphereH=320, ks=640, network="faster-rcnn", pretrained=True):
    bin_root = os.path.join(SphConv_ROOT, "caffemodel", "SphereH{0}Ks{1}".format(sphereH, ks))
    if not pretrained:
        bin_path = os.path.join(SphConv_ROOT, "caffemodel", "SphConv", "faster-rcnn5_3_iter_2048.caffemodel.h5")
        net.copy_from(bin_path)
        return net

    while layer != "1_1":
        if use_BatchNorm(layer):
            itr_fin = 4000
        else:
            itr_fin = 16000
        bin_dir = os.path.join(bin_root, "{0}{1}".format(network, layer))
        for tilt in xrange(sphereH):
            bin_path = os.path.join(bin_dir, "tilt{0:03d}_iter_{1}.caffemodel".format(tilt, itr_fin))
            if not os.path.isfile(bin_path):
                print "Skip {}".format(bin_path)
                continue
            print bin_path
            net.copy_from(bin_path)
        layer = top_down[layer]
    return net

def forward_sphconv(img, net, target=None):
    if isinstance(img, str):
        frameId = os.path.splitext(os.path.basename(img))[0]
        with h5py.File(img, 'r') as hf:
            frame = hf[frameId][:]
    elif isinstance(img, np.ndarray):
        frame = img
    else:
        raise ValueError("First argument has to be either string or ndarray.")
    frame = np.transpose(frame, (2,0,1))
    net.blobs['data'].data[...] = frame
    if target is None:
        target = net.blobs.keys()[-1]
    out = net.forward(blobs=[target])
    val = out[target][0,:,:,:].copy()
    val = np.transpose(val, (1, 2, 0))
    return val

