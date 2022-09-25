# This script applies random operations to all images in an input directory and saves the results in an output directory.
# The images are cropped and resized to 200x200 pixels and then compressed using JPEG at a random quality level.
#
# Usage:
#    python random_operations.py <input directory> <output directory> <seed>
#
# Libraries:
#    pillow=9.0.1
#    jpeg=9e
#    tqdm=4.63.0
#


import os
from PIL import Image
import tqdm
import shutil
import glob
from random import Random

output_size  = 200
cropsize_min = 160
cropsize_max = 2048
cropsize_ratio = (5,8)
qf_range = (65, 100)

def check_img(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))

def random_operations(input_dir, output_dir, seed, maximg=None):
    print('Random Operations from ', input_dir, 'to', output_dir, flush=True)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)  # remove existing output directory
    os.makedirs(output_dir)  # create output directory
    
    random = Random(seed) # set seed 
    
    # list fo images
    input_dir = input_dir if '*' in input_dir else os.path.join(input_dir, '*')
    list_src = [_ for _ in sorted(glob.glob(input_dir)) if check_img(_)]
    
    if maximg is not None:
        random.shuffle(list_src)  # shuffle the list of images
        list_src = list_src[:maximg]  # limit the number of images
    
    with open(os.path.join(output_dir, 'metainfo.csv'), 'w') as fid:
        fid.write('filename,src,cropsize,x1,y1,qf\n')
        for index, src in enumerate(tqdm.tqdm(list_src)):
            filename_dst = 'img%06d.jpg' % index
            dst = os.path.join(output_dir, filename_dst)
            
            # open image
            img = Image.open(src).convert('RGB')
            height = img.size[1]
            width = img.size[0]

            # select the size of crop
            cropmax = min(min(width, height), cropsize_max)
            if cropmax<cropsize_min:
                print(src, width, height)
            assert cropmax>=cropsize_min
            
            cropmin = max(cropmax*cropsize_ratio[0]//cropsize_ratio[1], cropsize_min)
            cropsize = random.randint(cropmin, cropmax)
            
            # select the type of interpolation
            interp = Image.ANTIALIAS if cropsize>output_size else Image.CUBIC
            
            # select the position of the crop
            x1 = random.randint(0, width - cropsize)
            y1 = random.randint(0, height - cropsize)
            
            # select the jpeg quality factor
            qf = random.randint(*qf_range)
            
            # make cropping
            img = img.crop((x1, y1, x1+cropsize, y1+cropsize))
            assert img.size[0]==cropsize
            assert img.size[1]==cropsize
            
            # make resizing
            img = img.resize((output_size, output_size), interp)
            assert img.size[0]==output_size
            assert img.size[1]==output_size
            
            # make jpeg compression
            img.save(dst, "JPEG", quality = qf)
            
            # save information
            fid.write(f'{filename_dst},{src},{cropsize},{x1},{y1},{qf}\n')
            
            
if __name__=='__main__':
    from sys import argv
    input_dir  = argv[1]
    output_dir = argv[2]
    seed       = int(argv[3])
    maximg     = int(argv[4]) if len(argv)>4 else None
    random_operations(input_dir, output_dir, seed, maximg)
    
