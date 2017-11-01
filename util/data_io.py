
import os
import sys
import cPickle as pickle

FRAME_ROOT = "SphericalConvolution/Frames"

class Logger(object):
    def __init__(self, log, stderr=False):
        if stderr:
            self.terminal = sys.stderr
        else:
            self.terminal = sys.stdout
        self.log = open(log, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def cleanup(self):
        self.log.close()
        self.terminal = None

def load_pkl(path):
    with open(path, 'rb') as fin:
        samples = pickle.load(fin)
    return samples

def dump_pkl(obj, path):
    if os.path.isfile(path):
        raise ValueError("{} exists!".format(path))
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout, protocol=2)

def get_frameId(path):
    frameId = os.path.splitext(os.path.basename(path))[0]
    return frameId

def collect_frames():
    frames = []
    filterd_frames = filter_frames(sample_rate=4)

    for c in os.listdir(FRAME_ROOT):
        cdir = os.path.join(FRAME_ROOT, c)
        for frame in os.listdir(cdir):
            if os.path.splitext(frame)[0] not in filterd_frames:
                continue
            path = os.path.join(cdir, frame)
            frames.append(path)
    return frames

def sample_pixels(tilt, sphereH=640):
    frames = collect_frames()
    n_samples = 40
    sphereW = sphereH * 2
    step = sphereW / n_samples
    xs = range(0, sphereW, step)

    samples = {}
    for frame in frames:
        pixels = []
        for x in xs:
            pixels.append((x, tilt))
        samples[frame] = pixels
    return samples

def filter_frames(sample_rate=2):
    frames = []
    for c in os.listdir(FRAME_ROOT):
        cdir = os.path.join(FRAME_ROOT, c)
        video_frames = {}
        for frame in os.listdir(cdir):
            videoId = frame[:11]
            frameId = frame[-9:-4]
            if not video_frames.has_key(videoId):
                video_frames[videoId] = []
            video_frames[videoId].append(frameId)
        for videoId, frameIds in video_frames.viewitems():
            frameIds = sorted(frameIds)
            frameIds = frameIds[::sample_rate]
            for frameId in frameIds:
                frame = "{0}-{1}".format(videoId, frameId)
                frames.append(frame)
    return frames

if __name__ == "__main__":
    frames = collect_frames()
    print len(frames)
