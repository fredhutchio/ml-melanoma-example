# IO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import glob
import json
from PIL import Image

# data manipulation, transformation
import numpy as np
from keras.applications.inception_v3 import preprocess_input as prep_iv3
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import datetime


# useful constants and collections
PREPROCESSORS = {'i': prep_iv3}
SHAPES = {'i': (299,299,3)}
INVALID_TARGET_STRINGS = ['/', 'other', 'unknown', 'indeterminate']


def preprocess_images(imgs, preprocessor, segs):
    '''
        Args:
        - imgs: list of numpy.ndarray images
        - preprocessor: str, indicates which architecture/processor to use
        - segs: optional list, corresponding numpy.ndarray segmentation images
        Return:
        - numpy.ndarray of preprocessed ndarray images
        - if segs is provided, each image is also reduced to its lesion area
    '''
    ret = []
    for i in range(len(imgs)):
        img = imgs[i]
       
        # segmentation step if needed
        if segs:
            idxs = np.where(segs[i] == [0])
            img[idxs] == np.array([0,0,0])
        
        # perform preprocessing step and append
        img = PREPROCESSORS[preprocessor](img)
        ret.append(img)
        
    # resize ret and return
    ret = np.array(ret)
    print('Preprocessed in mode %s, shape %s' % (preprocessor, ret.shape))
    return ret


def load_images(filepaths, shape, segs=False):
    '''
        Args:
        - filepaths: list of str paths to images
        - shape: (int, int) tuple, desired image shape
        - segs: boolean, indicates whether paths are segmentation files
        Return:
        - list of numpy.ndarray images with resizing
    '''
    imgs = [Image.open(x).resize(shape) for x in filepaths]
    
    # segmentation have only 1 channel indicating inside/outside of lesion
    if not segs:
        imgs = [x.convert('RGB') for x in imgs]
    
    # make images into ndarray and return
    ret = [np.array(x) for x in imgs]
    print('Loaded %d images' % len(ret))
    return ret

    
def load_targets(filepaths, response):
    '''
        Args:
        - filepaths: list of str paths to description files
        - response: str, indicates which attribute is target value
        Return:
        - list of int values representing encoded targets
        - indices of data with missing clinical attributes
    '''
    y, idxs = [], []
    for i in range(len(filepaths)):
        file = json.load(open(filepaths[i], 'r'))
        
        # try finding value
        try:
            data = file['meta']['clinical']
            data = data[response]
            
            # ensure value is valid string
            if data:
                if not any(inv in data for inv in INVALID_TARGET_STRINGS):
                    y.append(data)
                    continue
            
            # store is string invalid
            idxs.append(i)
                
        # store position if missing
        except KeyError:
            idxs.append(i)

    # encode targets
    le = LabelEncoder()
    le.fit(y)
    
    # transform, output, and return
    ret = le.transform(y), idxs
    print('Encoded targets into %d unique values' % len(le.classes_))
    return ret


def verify_ids(img_filepaths, dsc_filepaths, seg_filepaths):
    '''
        Args:
        - img_filepaths: list of str file paths to images
        - dsc_filepaths: list of str file paths to descriptions
        - seg_filepaths: optional list of str filepaths to segmentations
        Behavior:
        - assert: length of input lists is equal
        - assert: matching data points at each index of input lists
    '''    
    # match lengths
    num_img, num_dsc = len(img_filepaths), len(dsc_filepaths)
    assert num_img == num_dsc
    assert (len(seg_filepaths) == num_img) if seg_filepaths else True
    
    # matching elements at each index
    for i in range(num_img):
        img_id = img_filepaths[i].split('/')[-1].split('.')[0].split('_')[-1]
        dsc_id = dsc_filepaths[i].split('/')[-1].split('_')[-1]
        seg_id = None
        
        if seg_filepaths:
            seg_id = seg_filepaths[i].split('/')[-1].split('.')[0]
            seg_id = seg_id.split('_')[-2]
            
        assert img_id == dsc_id
        assert (img_id == seg_id) if seg_id else True
    
    # finish
    print('Verified %d points' % num_img)

    
def preprocess_data(img_dir='../imgs/',
                    mode='i',
                    dsc_dir='../dscs/',
                    response='benign_malignant',
                    seg_dir=None,
                    out_dir=None,
                    limit=None):
    '''
        Args:
        - img_dir: str, location of image files
        - mode: str, indicates which architecture to use
        - dsc_dir: str, location of description files
        - reponse: str, indicates target attribute
        - seg_dir: optional str, location of segmentation files
        - out_dir: optional str, location to save final results
        - limit: optional int, maximum data points to load
        Return:
        - X,y data as separate numpy.ndarray objects
        - if specified, this is written to disk
    '''
    img_filepaths = glob.glob(os.path.join(img_dir, '*'))
    dsc_filepaths = glob.glob(os.path.join(dsc_dir, '*'))
    seg_filepaths = glob.glob(os.path.join(seg_dir, '*')) if seg_dir else None
        
    # sort and validate 
    img_filepaths = sorted(img_filepaths)
    dsc_filepaths = sorted(dsc_filepaths)
    seg_filepaths = sorted(seg_filepaths) if seg_dir else None
    verify_ids(img_filepaths, dsc_filepaths, seg_filepaths)
    
    # for working with small samples
    if limit:
        img_filepaths = img_filepaths[0:limit]
        dsc_filepaths = dsc_filepaths[0:limit]
        seg_filepaths = seg_filepaths[0:limit] if seg_filepaths else None
    
    # check for segmentation data
    segs = None
    if seg_filepaths:
        segs = load_images(seg_filepaths, SHAPES[mode], segs=True)
    
    # get targets first in case of missing target values
    y, idxs = load_targets(dsc_filepaths, response=response)

    # load, preprocess
    X = load_images(img_filepaths, SHAPES[mode])
    X = preprocess_images(X, mode, segs=segs)
    
    # filter invalid rows if needed and shuffle
    mask = np.ones(X.shape[0], dtype=bool)
    mask[idxs] = False
    X = X[mask]
    X, y = shuffle(X, y)
    
    # write to disk if specified
    if out_dir:
        classif = 'binary' if response == 'benign_malignant' else 'multi'
        seg_lbl = 'original' if not seg_dir else 'segmented'
        session = '%s_%s_%s' % (mode, classif, seg_lbl)
        session = os.path.join(out_dir, session)
        np.save(session + '_X', X)
        np.save(session + '_y', y)
        print('Written to %s_X.npy and %s_y.npy' % (session, session))
        
    # output shapes and finish
    print('Data shapes are %s and %s' % (X.shape, y.shape))
    print('Done.')
    
    if not out_dir:
        return X,y
    
    
if __name__ == '__main__':
    fire.Fire(preprocess_data)