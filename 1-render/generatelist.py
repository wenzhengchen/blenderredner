

import os
import glob


trainlistname = './train_list.txt'

fd = open(trainlistname, 'w')

folder = '../1-render/render-re'


classes = glob.glob('%s/*'%folder)
for cls in classes:
    _, clsname = os.path.split(cls)
    
    models = glob.glob('%s/*'%cls)
    for model in models:
        _, modelname = os.path.split(model)
        
        ims = glob.glob('%s/*.png'%model)
        for im in ims:
            _, imname = os.path.split(im)
            imname, _ = os.path.splitext(imname)
            
            fd.write('%s/%s_%s_%s.dat'%(folder, clsname, modelname, imname))
            fd.write('\n')
            
fd.close()
        
        