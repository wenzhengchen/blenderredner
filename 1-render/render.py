

import os
import glob
# import numpy as np
import random
from multiprocessing import Pool


############################################3
viewnum = 24
shapenetfolder = './shapenet-v2'

cates = '04401088,02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04530566'
classes = cates.split(',')
classes = ['02958343']

renderfolder = './render-re'
if not os.path.exists(renderfolder):
    os.mkdir(renderfolder)

g_blender_path = '/Users/wenzhengchen/Downloads/blender-2.79b-macOS-10.6/blender.app/Contents/MacOS/blender'


for cls in classes:
    classfolder = '%s/%s' % (shapenetfolder, cls)
    modelfolder = glob.glob('%s/*' % classfolder)
    print(modelfolder)
    
    commands = []
    
    for modelfl in modelfolder:
        _, md5name = os.path.split(modelfl)
        syn_images_folder = os.path.join(renderfolder, cls, md5name) 
        
        # modelname = '%s/model-0.45.obj' % modelfl
        modelname = '%s/models/model_normalized-0.45.obj' % modelfl
        command = '%s ./blank.blend --background --python ./blendercall6.py -- %s %d %s' % (g_blender_path, modelname, viewnum, syn_images_folder);
        print(command)
        # os.system(command)
        commands.append(command)
    '''

for _ in range(1):
    
    file_list = '/scratch/gobi1/wenzheng/downloads/test_list.txt'
    pkl_list = []
    with open(file_list, 'r') as f:
        while(True):
            line = f.readline().strip()
            if not line:
                break
            pkl_list.append(line)
    pkl_list = sorted(pkl_list)
    pkl_list = [fl for fl in pkl_list if '00.dat' in fl]
    
    commands = []
    
    for modelfl in pkl_list:
        _, name = os.path.split(modelfl)
        cls, md5name, _ = name.split('_')
        syn_images_folder = os.path.join(renderfolder, cls, md5name) 
        
        # modelname = '%s/model-0.45.obj' % modelfl
        modelname = '%s/%s/%s/models/model_normalized-0.45.obj' % (shapenetfolder, cls, md5name)
        command = '%s ./blank.blend --background --python ./blendercall3.py -- %s %d %s' % (g_blender_path, modelname, viewnum, syn_images_folder);
        print(command)
        # os.system(command)
        commands.append(command)
    '''
    pool = Pool(processes=16)
    pool.map(os.system, commands)
    pool.close()
    pool.join()

