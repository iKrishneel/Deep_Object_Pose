#!/usr/bin/env python

import numpy as np
import cv2 as cv

import torch

import sys
sys.path.append('../src/dope/inference')

from detector import ModelData


def main(path_to_weight, test_im_path):
    model = ModelData()
    net = model.load_net_model_path(path_to_weight)

    im = cv.imread(test_im_path)
    im = cv.resize(im, (224, 224))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0

    x = torch.from_numpy(im).unsqueeze(0)
    y = net(x)

    print(len(y))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
