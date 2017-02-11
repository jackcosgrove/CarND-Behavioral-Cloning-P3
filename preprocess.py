import os
import numpy as np

def make_file_name_relative(path, file_name):
    start = file_name.find('/IMG')
    return path + file_name[start:]

def smooth(path, r, w):
    window_size = 6.0
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
            tokens[0] = make_file_name_relative(path, tokens[0])
            tokens[1] = make_file_name_relative(path, tokens[1])
            tokens[2] = make_file_name_relative(path, tokens[2])
            tokens[3] = str(sum(window) / window_size)
            w.write(str.join(',', tokens))
        i += 1

r = open("slalom/driving_log.csv", "r")
w = open("slalom/driving_log.smoothed.csv", "w")

smooth("slalom", r, w)

r.close()
w.close()

r = open("curves/driving_log.csv", "r")
w = open("curves/driving_log.smoothed.csv", "w")

smooth("curves", r, w)

r.close()
w.close()

def filter_slalom(r, w):
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
                    for i in range(int(right_len / 3)):
                        w.write(turning_right[i])
                    turning_right = []
            turning_left.append(line)
        elif angle > 0:
            if len(turning_right) == 0:
                left_len = len(turning_left)
                if left_len > 0:
                    for i in range(int(left_len / 3)):
                        w.write(turning_left[i])
                    turning_left = []
            turning_right.append(line)

r = open("slalom/driving_log.smoothed.csv", "r")
w = open("slalom/driving_log.filtered.csv", "w")

filter_slalom(r, w)

r.close()
w.close()

def split(r, t, v):
    for line in r:
        i = np.random.randint(4)
        if i == 0:
            v.write(line)
        else:
            t.write(line)


s = open("slalom/driving_log.filtered.csv", "r")
c = open("curves/driving_log.smoothed.csv", "r")

t = open("training/driving_log.training.csv", "w")
v = open("training/driving_log.validation.csv", "w")

split(s, t, v)
split(c, t, v)

s.close()
c.close()
t.close()
v.close()
