import os
import shutil
import numpy as np

def make_file_name_relative(path, file_name):
    start = file_name.find('/IMG')
    return path + file_name[start:]

def make_file_names_relative(path, line):
    tokens = line.split(',')
    tokens[0] = make_file_name_relative(path, tokens[0])
    tokens[1] = make_file_name_relative(path, tokens[1])
    tokens[2] = make_file_name_relative(path, tokens[2])
    return str.join(',', tokens)
    
def smooth(r, w):
    window_size = 4.0
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
            tokens[3] = str(sum(window) / window_size)
            w.write(str.join(',', tokens))
        i += 1

r = open("slalom/driving_log.csv", "r")
w = open("slalom/driving_log.smoothed.csv", "w")

smooth(r, w)

r.close()
w.close()

#r = open("curves/driving_log.csv", "r")
#w = open("curves/driving_log.smoothed.csv", "w")

#smooth("curves", r, w)

#r.close()
#w.close()

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
            if len(turning_left) > 0:
                turning_left.append(line)
            else:
                turning_right.append(line)

r = open("slalom/driving_log.smoothed.csv", "r")
w = open("slalom/driving_log.filtered.csv", "w")

filter_slalom(r, w)

r.close()
w.close()

def bin_index(value, bins):
    if value < bins[0]:
        return 0
    if value >= bins[-2]:
        return len(bins)-1 
    for i in range(len(bins)-1):
        if value >= bins[i] and value < bins[i+1]:
            return i
        
    
def distribute(data, w):
    data_len = len(data)
    angles = np.empty(data_len)
    for i in range(data_len):
        tokens = data[i].split(',')
        angles[i] = float(tokens[3].strip())

    hist, bins = np.histogram(angles, bins=50)
    bin_counts = np.zeros(51, dtype=np.int)
    min_bin_height = 20
    limits = (min_bin_height * np.abs(np.sin(bins * np.pi)) + min_bin_height).astype(np.int)
#    limits = np.ones(51) * 10
    i = 0
    misses = 0
    print("Total Samples: " + str(data_len))
    while i < limits.sum() and misses < 1000:
        line = data[np.random.randint(data_len)]
        tokens = line.split(',')
        angle = float(tokens[3].strip())
        idx = bin_index(angle, bins)
        if bin_counts[idx] < limits[idx]:
            i += 1
            if i % 500 == 0:
                print("Samples Acquired: " + str(i))
            bin_counts[idx] += 1
            w.write(line)
        else:
            misses += 1

r = open("slalom/driving_log.filtered.csv", "r")
w = open("slalom/driving_log.distributed.csv", "w")

data = r.readlines()
distribute(data, w)

r.close()
w.close()

def clear_images(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_image(file_path, destination):
    start = file_path.rfind('/')
    try:
        shutil.copyfile(file_path, destination + file_path[start:])
        return True
    except:
        return False

def copy_images(line, destination):
    tokens = line.split(',')
    success = copy_image(tokens[0].strip(), destination)
    success &= copy_image(tokens[1].strip(), destination)
    success &= copy_image(tokens[2].strip(), destination)
    return success
    
def copy_all_images(r, destination):
    clear_images(destination)
    for line in r:
        copy_images(line, destination)

def split(path, destination, r, t, v):
    for line in r:
        if not copy_images(line, destination):
            print("Warning: Missing images...")
            continue
        i = np.random.randint(4)
        if i == 0:
            v.write(make_file_names_relative(path, line))
        else:
            t.write(make_file_names_relative(path, line))


s = open("slalom/driving_log.distributed.csv", "r")
#c = open("curves/driving_log.smoothed.csv", "r")

t = open("training/driving_log.training.csv", "w")
v = open("training/driving_log.validation.csv", "w")

split("training", "training/IMG", s, t, v)
#split(c, t, v)

s.close()
#c.close()
t.close()
v.close()
