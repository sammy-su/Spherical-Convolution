
import os
import sys
import cv2
import h5py
import numpy as np

import cfg
import caffe

from collections import namedtuple
from multiprocessing import Process, Queue
from random import shuffle
from cfg import DATA_ROOT
from SphereProjection import crop_image
from util.data_io import load_pkl, get_frameId
from util.rf import rounded_rf, strides

class SphericalDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)

        # store input as class variables
        self.batch_size = params['batch_size']

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])

        # sample -> path, x, y, target
        samples = load_pkl(params['data_path'])
        NOch = len(samples[0][3])
        top[1].reshape(self.batch_size, NOch)

        patches = []
        targets = []
        crop_size = top[0].shape[2:]
        for path, x, y, target in samples:
            img = cv2.imread(path)
            patch = crop_image(img, x, y, crop_size)
            patch = patch - np.array([103.939, 116.779, 123.68])
            patch = np.transpose(patch, (2,0,1))
            patches.append(patch)

            target = np.array(target)
            targets.append(target)

        self.targets = np.vstack(targets)
        self.patches = np.stack(patches, axis=0)

        Ns = len(samples)
        self.samples = np.random.permutation(Ns)
        self._cur = 0
        self.Ns = Ns

    def forward(self, bottom, top):
        """
        Load data.
        """
        for i in range(self.batch_size):
            idx = self.samples[self._cur]
            top[0].data[i, ...] = self.patches[idx]
            top[1].data[i, ...] = self.targets[idx]

            self._cur += 1
            if self._cur == len(self.samples):
                self._cur = 0
                self.samples = np.random.permutation(self.Ns)

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class EuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        if len(bottom[0].data.shape) == 4:
            assert bottom[0].data.shape[2] == 1
            assert bottom[0].data.shape[3] == 1
            self.diff = bottom[0].data - bottom[1].data[:,:,np.newaxis,np.newaxis]
        elif len(bottom[0].data.shape) == 2:
            self.diff = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].count / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].count


class SphericalIntermediateLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)

        # store input as class variables
        self.batch_size = params['batch_size']
        data_path = params['data_path']

        ext = os.path.splitext(data_path)[-1]
        if ext == ".h5":
            with h5py.File(data_path, 'r') as hf:
                self.targets = hf['target'][:]
                srcs = hf['srcs'][:]
                INDn = srcs < 0.
                srcs[INDn] = 0.
                self.srcs = srcs
        elif ext == ".pkl":
            data = load_pkl(data_path)
            targets = []
            srcs = []
            if len(data[0]) == 5:
                for _, _, _, target, src in data:
                    target = np.array(target)
                    targets.append(target)

                    INDn = src < 0
                    src[INDn] = 0.
                    srcs.append(src)
                self.targets = np.vstack(targets)
                self.srcs = np.stack(srcs, axis=0)
            else:
                raise ValueError("Only supported cropped input.")
        else:
            raise ValueError("Invalid input data type ({})".format(data_path))
        NIch = self.srcs.shape[3]
        NOch = self.targets.shape[1]
        Ns = self.targets.shape[0]

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, NIch, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self.batch_size, NOch)

        self.samples = np.random.permutation(Ns)
        self._cur = 0
        self.Ns = Ns

    def forward(self, bottom, top):
        """
        Load data.
        """
        for i in range(self.batch_size):
            idx = self.samples[self._cur]
            self._cur += 1
            if self._cur == self.Ns:
                self._cur = 0
                self.samples = np.random.permutation(self.Ns)

            target = self.targets[idx]
            src = self.srcs[idx]
            src = np.transpose(src, (2,0,1))
            top[0].data[i, ...] = src
            top[1].data[i, ...] = target

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass



Row = namedtuple('Row', ['Id', 'rowId', 'shift'])

class SplitRowLayer(caffe.Layer):

    def setup(self, bottom, top):
        # === Read input parameters ===
        params = eval(self.param_str)

        batch_size, Nch, H, W = bottom[0].data.shape
        layer = params['layer']
        sphereH = params.get('sphereH', 320)
        ks = params.get('ks', 640)
        stride = strides[ks][sphereH][layer]

        assert len(top) == H
        assert sphereH / H == stride

        # === reshape tops ===
        top_rows = {}
        pads = {}
        for i in xrange(H):
            tilt = i * stride
            _, (kernel_w, kernel_h) = rounded_rf(layer, tilt, sphereH=sphereH, ks=ks)
            pad = kernel_w / 2
            crop_w = W + 2 * pad
            top[i].reshape(batch_size, Nch, kernel_h, crop_w)

            pads[i] = pad
            rows = []
            for j in xrange(kernel_h):
                rowId = i + j - kernel_h / 2
                if rowId < 0:
                    rowId = -1 - rowId
                    shift = True
                elif rowId >= H:
                    rowId = H - 1 - (rowId - H)
                    shift = True
                else:
                    shift = False
                row = Row(j, rowId, shift)
                rows.append(row)
            top_rows[i] = rows
        self.top_rows = top_rows
        self.pads = pads

    def forward(self, bottom, top):
        """
        Load data.
        """
        batch_size, Nch, H, W = bottom[0].data.shape
        for i in range(H):
            rows = self.top_rows[i]
            pad = self.pads[i]
            for row in rows:
                row_data = bottom[0].data[:,:,row.rowId:row.rowId+1,:]
                if row.shift:
                    row_data = np.roll(row_data, W/2, axis=3)
                top[i].data[:,:,row.Id:row.Id+1,pad:-pad] = row_data
                top[i].data[:,:,row.Id:row.Id+1,:pad] = row_data[:,:,:,-pad:]
                top[i].data[:,:,row.Id:row.Id+1,-pad:] = row_data[:,:,:,:pad]

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        batch_size, Nch, H, W = bottom[0].data.shape
        bottom[0].diff.fill(0)
        for i in range(len(top)):
            rows = self.top_rows[i]
            pad = self.pads[i]

            top_data = top[i].diff[:,:,:,pad:-pad]
            top_data[:,:,:,:pad] += top[i].diff[:,:,:,-pad:]
            top_data[:,:,:,-pad:] += top[i].diff[:,:,:,:pad]

            for row in rows:
                row_data = top_data[:,:,row.Id:row.Id+1,:]
                if row.shift:
                    row_data = np.roll(row_data, -W/2, axis=3)
                bottom[0].diff[:,:,row.rowId:row.rowId+1,:] += row_data

class MergeRowLayer(caffe.Layer):

    def setup(self, bottom, top):
        H = len(bottom)
        batch_size, Nch, _, W = bottom[0].data.shape
        top[0].reshape(batch_size, Nch, H, W)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for i in xrange(len(bottom)):
            top[0].data[:,:,i:i+1,:] = bottom[i].data[...]

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = top[0].diff[:,:,i:i+1,:]


class SphConvDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # === Read input parameters ===
        params = eval(self.param_str)

        layer = params['layer']
        sphereH = params.get('sphereH', 320)
        sphereW = sphereH * 2
        ks = params.get('ks', 640)
        network = params.get('network', 'faster-rcnn')
        phase = params.get('phase', 'train')
        prefetch = params.get('prefetch', 1)
        stride = strides[ks][sphereH][layer]
        self.stride = stride
        H = sphereH / stride
        W = H * 2

        self.sphereH = sphereH
        self.ks = ks
        self.network = network
        self.layer = layer

        target_path = os.path.join(DATA_ROOT,
                                   "TargetSphereH{0}Ks{1}".format(sphereH, ks),
                                   "{0}{1}".format(network, layer),
                                   "tilt{0:03d}.{1}.pkl".format(sphereH/2, phase))
        data = load_pkl(target_path)
        samples = list(set(get_frameId(path) for path, _, _, _ in data))
        self.samples = samples

        self.prefetch = prefetch
        if prefetch:
            self.setup_prefetch()
        else:
            shuffle(self.samples)
            self._cur = 0

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        batch_size = 1
        Nch = len(data[0][3])
        top[0].reshape(batch_size, 64, sphereH, sphereW)
        top[1].reshape(batch_size, Nch, H, W)

    def fetch_data(self):
        frameId = self.samples[self._cur]
        self._cur += 1
        if self._cur == len(self.samples):
            shuffle(self.samples)
            self._cur = 0

        print frameId
        src_path = os.path.join(DATA_ROOT,
                               "SourceSphereH{0}Ks{1}".format(self.sphereH, self.ks),
                               "{0}1_1".format(self.network),
                               "{}.h5".format(frameId))
        with h5py.File(src_path, 'r') as hf:
            src = hf[frameId][:]
        label_path = os.path.join(DATA_ROOT,
                                 "SourceSphereH{0}Ks{1}".format(self.sphereH, self.ks),
                                 "{0}{1}".format(self.network, self.layer),
                                 "{}.h5".format(frameId))
        with h5py.File(label_path, 'r') as hf:
            label = hf[frameId][:]
        return src, label

    def forward(self, bottom, top):
        """
        Load data.
        """
        if self.prefetch:
            src, label = self.prefetch_data()
        else:
            src, label = self.fetch_data()
        src = np.transpose(src, (2,0,1))
        if self.stride > 1:
            label = label[::self.stride,::self.stride,:]
        label = np.transpose(label, (2,0,1))
        top[0].data[0,...] = src
        top[1].data[0,...] = label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

    def setup_prefetch(self):
        n_process = self.prefetch
        max_queue_size = n_process * 4

        sample_q = Queue()
        pid_q = Queue()
        jobs = np.random.permutation(len(self.samples))
        for job in jobs:
            sample_q.put(job)
        data_q = Queue(max_queue_size)

        procs = []
        for p in xrange(n_process):
            proc = PrefetchProcess(self.samples, sample_q, data_q, pid_q,
                                   layer=self.layer,
                                   sphereH=self.sphereH,
                                   ks=self.ks,
                                   network=self.network,
                                   root_dir=DATA_ROOT)
            proc.start()
            procs.append(proc)
        self.procs = procs
        self.sample_q = sample_q
        self.data_q = data_q
        self.pid_q = pid_q
        self.n_process = n_process

    def prefetch_data(self):
        if self.sample_q.qsize() < self.n_process:
            jobs = np.random.permutation(len(self.samples))
            for job in jobs:
                self.sample_q.put(job)
        src, label = self.data_q.get()
        return src, label

    def __del__(self):
        self.sample_q.put(-1)
        #for i in xrange(self.n_process):
        #    pid = self.pid_q.get()
        #    os.kill(pid, signal.SIGTERM)
        for proc in self.procs:
            proc.terminate()
            proc.join()


class SphConvLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        if len(bottom) == 3:
            assert bottom[0].shape[2] == bottom[2].shape[0]
            assert bottom[0].shape[3] == bottom[2].shape[1]
            for i in xrange(bottom[0].data.shape[2]):
                for j in xrange(bottom[0].data.shape[3]):
                    if not bottom[2].data[i,j]:
                        self.diff[:,:,i,j] = 0.
            Nch = bottom[0].shape[1]
            Nvalid = np.sum(bottom[2].data)
            N = Nch * Nvalid
        elif len(bottom) == 2:
            N = bottom[0].count
        else:
            raise ValueError("Need two/three input for loss.")
        top[0].data[...] = np.sum(self.diff**2) / N / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].count


class PrefetchProcess(Process):
    def __init__(self, frameIds, sample_q, data_q, pid_q, **kwargs):
        super(PrefetchProcess, self).__init__()
        self.frameIds = frameIds
        self.sample_q = sample_q
        self.data_q = data_q
        self.pid_q = pid_q

        self.layer = kwargs['layer']
        self.sphereH = kwargs.get('sphereH', 320)
        self.ks = kwargs.get('ks', 640)
        self.network = kwargs.get('network', 'faster-rcnn')
        self.root_dir = kwargs.get('root_dir', DATA_ROOT)

    def check_parent(self):
        try:
            os.kill(self.parent_pid, 0)
        except:
            sys.exit()

    def run(self):
        pid = os.getpid()
        self.pid_q.put(pid)

        self.parent_pid = os.getppid()
        while True:
            self.check_parent()
            job = self.sample_q.get()
            if job < 0:
                self.sample_q.put(job)
                break
            frameId = self.frameIds[job]
            src_path, label_path = self.build_path(frameId)
            with h5py.File(src_path, 'r') as hf:
                src = hf[frameId][:]
            with h5py.File(label_path, 'r') as hf:
                label = hf[frameId][:]
            self.data_q.put((src, label))

    def build_path(self, frameId):
        src_dir = os.path.join(self.root_dir,
                               "SourceSphereH{0}Ks{1}".format(self.sphereH, self.ks),
                               "{0}1_1".format(self.network))
        label_dir = os.path.join(self.root_dir,
                                 "SourceSphereH{0}Ks{1}".format(self.sphereH, self.ks),
                                 "{0}{1}".format(self.network, self.layer))
        src_path = os.path.join(src_dir, "{}.h5".format(frameId))
        label_path = os.path.join(label_dir, "{}.h5".format(frameId))
        return src_path, label_path
