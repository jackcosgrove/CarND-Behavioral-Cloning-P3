import sys
import numpy as np

def process_line(line):
    tokens = line.split(',')
    file_name = tokens[0]
    start = file_name.find('/IMG')
    tokens[0] = file_name[start:]
    return str.join(',', tokens)

file_name = sys.argv[1] + '/driving_log.csv'
r = open(file_name, 'r')
t = open(file_name + '.training', 'w')
v = open(file_name + '.validation', 'w')

for line in r:
    i = np.random.randint(4)
    if i == 0:
        v.write(line)
    else:
        t.write(line)
r.close()
t.close()
v.close()
