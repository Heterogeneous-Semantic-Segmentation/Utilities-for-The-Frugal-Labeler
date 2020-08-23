import glob
from PIL import Image
from pathlib import Path
import os
from optparse import OptionParser

# Parse Options from Command Line
parser = OptionParser()
parser.add_option("--s", "--sourcedir", dest="source_dir", help="The source dir (non binary images)")
parser.add_option("--sf", "--sourcefalsevalue", dest="source_false_value", help="The source false value as a three tuple (default: (0,0,0))", default=(0,0,0))
parser.add_option("--tf", "--targetfalsevalue", dest="target_false_value", help="The target false value as a three tuple (default: (0,0,0))",  default=(0,0,0))
parser.add_option("--tt", "--targettruevalue", dest="target_true_value", help="The target true value as a three tuple (default: (255,255,255))", default=(255, 255, 255))
(options, args) = parser.parse_args()

if(options.source_dir == None):
    print('No source dir given. Exiting.')
    exit(-1)

img_dir = options.source_dir
target_dir = img_dir.rstrip("/")+'_binary/'
Path(target_dir).mkdir(parents=True, exist_ok=True)
source_false_value = options.source_false_value
target_false_value = options.target_false_value
target_true_value = options.target_true_value

for img_path in glob.glob(img_dir+'/*'):
    img_old = Image.open(img_path)
    filename = os.path.basename(img_path)
    img_new = Image.new(img_old.mode, img_old.size)
    pixel_map = img_old.load()
    pixels_new = img_new.load()
    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            if pixel_map[i,j] == source_false_value:
                pixels_new[i, j] = target_false_value
            else:
                pixels_new[i, j] = target_true_value
    img_new.save(target_dir+filename)
