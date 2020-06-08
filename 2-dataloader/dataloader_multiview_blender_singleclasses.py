

from __future__ import print_function
from __future__ import division

import os
import glob

import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from utils.utils_mesh import loadobj
from utils.utils_perspective import camera_info, perspectiveprojectionnp


#######################################################
class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, file_list, \
                 imszs=[-1], \
                 viewnum=1, \
                 catname='all', \
                 mode='train', \
                 folder='../1-render/render-re', \
                 camfolder='../1-render/render-re', \
                 modelfolder='../1-render/shapenet-v2', \
                 datadebug=False):
        """
        split: 'train', 'train_val' or 'val'
        """
        self.mode = mode
        self.datadebug = datadebug
        
        self.imszs = imszs
        self.viewum = viewnum
        assert self.viewum >= 1
        
        ##########################
        self.folder, _ = os.path.split(file_list)
        self.camfolder = '%s/data/ShapeNetRendering' % (self.folder)
        
        self.folder = folder
        self.camfolder = camfolder
        self.modelfolder = modelfolder
        
        self.pkl_list = []
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        
        if catname != 'all':
            self.pkl_list = [i for i in self.pkl_list if catname in i]
        
        if mode == 'test':
            ranidx = np.random.permutation(len(self.pkl_list))
            self.pkl_list = [self.pkl_list[i] for i in ranidx]
        
        self.imnum = len(self.pkl_list)
        print(self.pkl_list[0])
        print(self.pkl_list[-1])
        print('imnum {}'.format(self.imnum))
    
    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    
    def load_im_cam(self, pkl_path, catagory, md5name, num):
        
        imname = '%s/%s/%s/%d.png' % (self.folder, catagory, md5name, num)
        img = cv2.imread(imname, cv2.IMREAD_UNCHANGED)
        
        ims = []
        for imsz in self.imszs:
            if imsz != 256:
                im = cv2.resize(img, (imsz, imsz))
            else:
                im = img
            im_hxwx4 = im.astype('float32') / 255.0
            ims.append(im_hxwx4)
        
        if self.datadebug:
            cv2.imshow("imgt1", im_hxwx4[:, :, :3])
            cv2.imshow("immask", im_hxwx4[:, :, 3])
            # cv2.imshow("imgt2", im_hxwx4_old[:, :, :3])
            # cv2.imshow("imdiff", np.abs(im_hxwx4 - im_hxwx4_old)[:, :, :3])
            cv2.waitKey(0)
        
        rotntxname = '%s/%s/%s/%d.npy' % (self.camfolder, catagory, md5name, num)
        rotmtx_4x4 = np.load(rotntxname).astype(np.float32)
        rotmx = rotmtx_4x4[:3, :3]
        transmtx = rotmtx_4x4[:3, 3:4]
        transmtx = -np.matmul(rotmx.T, transmtx)
        renderparam = (rotmx, transmtx)
        
        if self.datadebug:
            
            '''
            renderparamname = '%s/%s/%s/cameras.txt' % (self.camfolder, catagory, md5name)
            renderparams = np.loadtxt(renderparamname, dtype=np.float32)
            renderparam = renderparams[num]
            rot2, trans2 = camera_info(renderparam)
            '''
            
            modelname = '%s/%s/%s/models/model_normalized-0.45.obj' % (self.modelfolder, catagory, md5name)
            position, _ = loadobj(modelname)
            print('max, {}, min, {}, biggest, {}'.format(np.max(position, 0), np.min(position, 0), np.max(np.abs(position))))
            
            pt_trans = np.matmul(position - transmtx.T, rotmx[:3, :3].T)
            X, Y, Z = pt_trans.T
            
            fovy = 49.13434207744484 / 180.0 * 3.141592653589793238463
            per_mtx = perspectiveprojectionnp(fovy, ratio=1.0, near=0.1, far=100.0)
            
            x = X / -Z * per_mtx[0, 0]
            y = Y / -Z * per_mtx[1, 0]
            
            im = cv2.resize(img[:, :, :3], (256, 256))
            h, w = 256, 256
            for i, xy in enumerate(zip(x, y)):
                x, y = xy
                x = (x + 1) / 2
                y = (-y + 1) / 2
                x = int(x * w)
                y = int(y * h)
                cv2.circle(im, (x, y), 3, (55, 255, 155), 1)
                
            cv2.imshow('', im)
            cv2.waitKey()
        
        return ims, renderparam
    
    def prepare_instance(self, idx):
        """
        Prepare a single instance
        """
        
        re = {}
        re['valid'] = True
        
        # name parameters
        pkl_path = self.pkl_list[idx]
        _, fname = os.path.split(pkl_path)
        fname, _ = os.path.splitext(fname)
        catagory, md5name, numname = fname.split('_')
        re['cate'] = catagory
        re['md5'] = md5name
        
        try:
            if self.viewum == 1:
                num = int(numname)
                ims, renderparam = self.load_im_cam(pkl_path, catagory, md5name, num)
                
                i = 0
                re['view%d' % i] = {}
                re['view%d' % i]['camrot'] = renderparam[0]
                re['view%d' % i]['campos'] = renderparam[1]
                re['view%d' % i]['num'] = num
                for imi, imsz in enumerate(self.imszs):
                    re['view%d' % i]['im%d' % imsz] = ims[imi]
            else:
                for i in range(self.viewum):
                    # 24 views in total
                    num = np.random.randint(24)
                        
                    ims, renderparam = self.load_im_cam(pkl_path, catagory, md5name, num)
                    
                    re['view%d' % i] = {}
                    re['view%d' % i]['camrot'] = renderparam[0]
                    re['view%d' % i]['campos'] = renderparam[1][:, 0]
                    re['view%d' % i]['num'] = num
                    for imi, imsz in enumerate(self.imszs):
                        re['view%d' % i]['im%d' % imsz] = ims[imi]
        except:
            re['valid'] = False
            return re
        
        return re


def collate_fn(batch_list):
    for data in batch_list:
        if not data['valid']:
            print('{}, {}'.format(data['cate'], data['md5']))
    
    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    if len(batch_list) == 0:
        return None
    
    # keys = batch_list[0].keys()
    keys = ['cate', 'md5']
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val
    
    viewnum = len(batch_list[0].keys()) - 3
    keys = ['im64', 'im256', 'camrot', 'campos', 'num']
    for i in range(viewnum):
        collated['view%d' % i] = {}
        for key in keys:
            val = [item['view%d' % i][key] for item in batch_list]
            val = np.stack(val, axis=0)
            collated['view%d' % i][key] = val

    return collated


def get_data_loaders(filelist, imszs, viewnum, catname, mode, bs, numworkers):
    
    print('Building dataloaders')
    
    dataset_train = DataProvider(filelist, imszs, viewnum, catname, \
                                 mode=mode, datadebug=False)
    
    shuffle = True
    if mode == 'train_val' or mode == 'test':
        shuffle = False
    
    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)
    
    print('train num {}'.format(len(dataset_train)))
    print('train iter'.format(len(train_loader)))
    
    return train_loader


##############################################
if __name__ == '__main__':
    
    file_folder = '../1-render'
    file_list = 'train_list.txt'
 
    imszs = [64, 256]
    viewnum = 2
    catname = '02958343'

    train_loader = get_data_loaders('%s/%s' % (file_folder, file_list), \
                                    imszs, viewnum, catname, mode='train', \
                                    bs=32, numworkers=0)
    
    ##############################################
    for i, data in enumerate(train_loader):
        if data is None:
            continue
        for j in range(viewnum):
            for key in ['im64', 'im256', 'camrot', 'campos', 'num']:
                print('{}, view{}, {}, {}'.format(i, j, key, data['view%d' % j][key].shape))
        
        for key in ['cate', 'md5']:
            print('{}, {}, {}'.format(i, key, data[key]))

