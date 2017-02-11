import os
import numpy as np

def make_file_name_relative(file_name):
    start = file_name.find('IMG')
    return file_name[start:]

r = open("training/driving_log.csv", "r")
w = open("training/driving_log.smoothed.csv", "w")

window_size = 12.0
lag_index = int(window_size / 2)
window = []
win_len = 0
lines = r.readlines()
i = 0
for line in lines:
    tokens = line.split(',')
    angle = float(tokens[3].strip())
    window.append(angle)
    win_len = len(window)
    if win_len > window_size:
        window.pop(0)
        to_add = lines[i-lag_index]
        tokens = to_add.split(',')
        tokens[0] = make_file_name_relative(tokens[0])
        tokens[1] = make_file_name_relative(tokens[1])
        tokens[2] = make_file_name_relative(tokens[2])
        tokens[3] = str(sum(window) / window_size)
        w.write(str.join(',', tokens))
    i += 1

r.close()
w.close()

r = open("training/driving_log.smoothed.csv", "r")
w = open("training/driving_log.filtered.csv", "w")

turning_left = []
turning_right = []

lines = r.readlines()
for line in lines:
    tokens = line.split(',')
    angle = float(tokens[3].strip())
    if angle < 0:
        if len(turning_left) == 0:
            right_len = len(turning_right)
            if right_len > 0:
                for i in range(int(right_len / 2)):
                    w.write(turning_right[i])
                turning_right = []
        turning_left.append(line)
    elif angle > 0:
        if len(turning_right) == 0:
            left_len = len(turning_left)
            if left_len > 0:
                for i in range(int(left_len / 2)):
                    w.write(turning_left[i])
                turning_left = []
        turning_right.append(line)
    else:
        if os.path.isfile(tokens[0]):
            os.remove(tokens[0])
        if os.path.isfile(tokens[1]):
            os.remove(tokens[1])
        if os.path.isfile(tokens[2]):
            os.remove(tokens[2])

r.close()
w.close()

r = open("training/driving_log.filtered.csv", "r")
t = open("training/driving_log.training.csv", "w")
v = open("training/driving_log.validation.csv", "w")

for line in r:
    i = np.random.randint(4)
    if i == 0:
        v.write(line)
    else:
        t.write(line)
r.close()
t.close()
v.close()
