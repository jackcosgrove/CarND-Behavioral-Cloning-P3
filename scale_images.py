import sys
import cv2
import matplotlib.image as mpimg
import scipy.misc

def process_line(line):
    tokens = line.split(',')
    img = mpimg.imread(tokens[0])
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    scipy.misc.imsave(tokens[0], img)

file_name = sys.argv[1]
f = open(file_name)
for line in f:
    process_line(line)
f.close()