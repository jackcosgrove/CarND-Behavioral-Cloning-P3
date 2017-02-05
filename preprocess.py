import sys
import cv2
import matplotlib.image as mpimg
import scipy.misc

def process_line(line):
    tokens = line.split(',')
    file_name = tokens[0]
    img = mpimg.imread(file_name)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    scipy.misc.imsave(file_name, img)
    start = file_name.find('/IMG')
    tokens[0] = file_name[start:]
    return str.join(',', tokens)

file_name = sys.argv[1]
r = open(file_name, 'r')
w = open(file_name + '.relative', 'w')
for line in r:
    w.write(process_line(line))
r.close()
w.close()