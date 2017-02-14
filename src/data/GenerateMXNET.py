#! /bin/python
from __future__ import print_function
import os
import random
import sys
import time


def check_path_or_fail(path, is_file=True):
    if os.path.exists(path):
        if is_file:
            if os.path.isfile(path):
                return True
            else:
                print("path %s not a valid file, exit" % path)
                exit(1)
        if not is_file:
            if os.path.isdir(path):
                return True
            else:
                print("path %s not a valid folder, exit" % path)
                exit(1)


def get_timestamp_str():
    return str(int(time.time()))


def create_image_tree(data_path_, data_file_path_):
    line = "tree -i -f %s |grep \".JPEG\|.png\|.bmp\" > %s" % (data_path_, data_file_path_)
    os.system(line)


def get_image_list_from_file(file_name):
    f = open(file_name, "rb")
    lines = [line.strip() for line in f.readlines()]
    return lines


def get_class_name_from_image_list(image_list, data_path_):
    class_list = []
    for image_ in image_list:
        class_name = image_.replace(data_path_, "").split("/")[0]
        class_list.append(class_name)
    return list(set(class_list))


def split_image_file_list_into_train_and_val(image_file_list, class_name_list, train_ratio=0.8, count_filter_=0,
                                             image_limits=None):
    train_set = []
    val_set = []
    label_set = []
    label_id = 0
    index = 0
    for className in class_name_list:
        search_key = "/" + className + "/"
        current_class_list = []
        for image_ in image_file_list:
            if image_.find(search_key) >= 0:
                current_class_list.append(image_)
        if len(current_class_list) < count_filter_:
            # skip small image_ set.
            pass
        else:
            label_set.append((label_id, className))
            random.shuffle(current_class_list)
            if image_limits is None:
                size_train = int(len(current_class_list) * train_ratio)
                for item in current_class_list[:size_train]:
                    train_set.append((index, label_id, item))
                    index += 1
                for item in current_class_list[size_train:]:
                    val_set.append((index, label_id, item))
                    index += 1
            else:
                size_train = int(image_limits * train_ratio)
                for item in current_class_list[:size_train]:
                    train_set.append((index, label_id, item))
                    index += 1
                for item in current_class_list[size_train:image_limits]:
                    val_set.append((index, label_id, item))
                    index += 1
            label_id += 1
    return train_set, val_set, label_set


def write_down_to_file(data_set, file_name):
    ft = open(file_name, "w")
    if len(data_set[0]) == 2:
        for line in data_set:
            wline = str(line[0]) + " \t " + str(line[1]) + "\n"
            ft.write(wline)
    else:
        for line in data_set:
            wline = str(line[0]) + " \t " + str(line[1]) + " \t " + str(line[2]) + "\n"
            ft.write(wline)
    ft.flush()
    ft.close()


if __name__ == "__main__":
    # print('sys.argv length', len(sys.argv))
    # print('sys.argv[0]', sys.argv[0])
    # print('sys.argv[1]', sys.argv[1])
    # print('sys.argv[2]', sys.argv[2])
    # print('sys.argv[3]', sys.argv[3])
    # print('sys.argv[4]', sys.argv[4])
    # print('sys.argv[5]', sys.argv[5])

    # exit(1)
    start = time.time()
    timestamp = get_timestamp_str()
    MXNETHome = os.getenv("MXNET_HOME", "/home/kiril/mxnet")
    if MXNETHome is None:
        print("MXNET_HOME is not defined!")
        exit(1)

    # if len(sys.argv) < 4:
    #     print("Usage: python %s data_path train_out.bin val_out.bin shape [ratio count_filter limits]")
    #     exit(1)
    if len(sys.argv) > 1:
        data_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) > 2:
        train_path = os.path.abspath(sys.argv[2])
    if len(sys.argv) > 3:
        val_path = os.path.abspath(sys.argv[3])

    # data_path = os.path.abspath(sys.argv[1])
    # train_path = os.path.abspath(sys.argv[2])
    # val_path = os.path.abspath(sys.argv[3])

    data_path = '../../data/raw/'
    train_path = '../../data/processed/scoodit_178_train.rec'
    val_path = '../../data/processed/scoodit_178_test.rec'
    check_path_or_fail(data_path, False)
    print("Data Path check PASS.")

    shape = 224
    ratio = 0.95
    count_filter = 0
    limits = None

    if len(sys.argv) > 4:
        shape = int(sys.argv[4])
    if len(sys.argv) > 5:
        ratio = float(sys.argv[5])
    if len(sys.argv) > 6:
        count_filter = int(sys.argv[6])
    if len(sys.argv) > 7:
        limits = int(sys.argv[7])
        ##################################################################
    prefix = '../../data/processed/'
    data_file_path = os.path.abspath(prefix + 'data.txt')
    labels_file_path = os.path.abspath(prefix + 'labels.txt')
    train_file_path = os.path.abspath(prefix + 'train.txt')
    val_file_path = os.path.abspath(prefix + 'val.txt')

    create_image_tree(data_path, data_file_path)
    imageFileList = get_image_list_from_file(data_file_path)
    # remove data_path in beginning
    imageFileList = [image.replace(os.path.abspath(data_path), "") for image in imageFileList]
    classNameList = get_class_name_from_image_list(imageFileList, data_path)
    trainSet, valSet, labelSet = split_image_file_list_into_train_and_val(imageFileList, classNameList, ratio,
                                                                          count_filter, limits)
    random.shuffle(trainSet)
    write_down_to_file(trainSet, train_file_path)
    write_down_to_file(valSet, val_file_path)
    write_down_to_file(labelSet, labels_file_path)

    ##################################################################
    im2rec = os.path.join(MXNETHome, "bin/im2rec")
    command = "%s %s %s %s resize=%s" % (im2rec, train_file_path, data_path, train_path, shape)
    print(command)
    os.system(command)

    command = "%s %s %s %s resize=%s" % (im2rec, val_file_path, data_path, val_path, shape)
    print(command)
    os.system(command)

    end = time.time()
    print("Time Elapsed: %.3fs." % (end - start))
