__author__ = 'ninaoa1'


from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)

from coco_to_hdf5_data import *
from captioner import Captioner

COCO_EVAL_PATH = './data/coco/coco-caption-eval'
sys.path.append(COCO_EVAL_PATH)
sys.path.insert(0, './python')

from pycocoevalcap.eval import COCOEvalCap
import caffe
import matplotlib.pyplot as plt

ITER = 50000
MODEL_FILENAME = 'lrcn_finetune_iter_%d' % ITER

MODEL_DIR = './examples/coco_caption'
MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
IMAGE_NET_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
LSTM_NET_FILE = './examples/coco_caption/lrcn_word_to_preds.deploy.prototxt'


VOCAB_FILE = './examples/coco_caption/h5_data/buffer_100/vocabulary.txt'
DEVICE_ID = 0

captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE, device_id=DEVICE_ID)


# IMAGE_FILE = '/files/vol07/LSEL/dev/linux/caffe/rc2/examples/images/fish-bike.jpg'
# IMAGE_FILE = '/files/vol07/LSEL/dev/linux/caffe/rc2/examples/images/cat.jpg'
# IMAGE_FILE = '/files/vol07/LSEL/dev/linux/caffe/rc2/examples/images/t72_tank.jpg'


directory = './data/coco/coco/images/val2014/'
files = os.listdir(directory)

version = 'oliver' #This is the improved version. It is slower and needs some work. You will see the difference if you let it run
#on several images

# version = 'standard' #The standard version is much faster but sometimes is erroneous

for IMAGE_FILE in files:
    IMAGE_FILE= directory+IMAGE_FILE
    input_image = caffe.io.load_image(IMAGE_FILE)
    images = [IMAGE_FILE]
    descriptor_filename = 'retrieval_cache/descriptors.npz'
    descriptors = captioner.compute_descriptors(images)
    np.savez_compressed(descriptor_filename, descriptors=descriptors)

    if version == 'standard':
        output_captions, output_probs = captioner.sample_captions(descriptors)
        caption = captioner.sentence(output_captions[0])
    elif version == 'oliver':
        print 'exploring trees'
        output_captions, output_probs = captioner.sample_captions2(descriptors,temp=float('inf'))#,temp=float('inf')
        caption = output_captions

    fig = plt.figure()
    fig.suptitle(caption, fontsize=11)
    plt.imshow(input_image)
    # plt.savefig('results/'+IMAGE_FILE.split('/')[-1]) #Uncomment to save results to a directory
    plt.show()
