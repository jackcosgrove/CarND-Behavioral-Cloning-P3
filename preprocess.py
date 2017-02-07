import sys

def process_line(line):
    tokens = line.split(',')
    file_name = tokens[0]
    start = file_name.find('/IMG')
    tokens[0] = file_name[start:]
    return str.join(',', tokens)

file_name = sys.argv[1]
r = open(file_name, 'r')
w = open(file_name + '.relative', 'a')
for line in r:
    w.write(process_line(line))
r.close()
w.close()
