import tensorflow as tf
import random
import os
import numpy as np
import tensorflow.contrib.slim as slim
import time
import logging
from PIL import Image

fromPath = "C:\\Users\\lele.chen\\Downloads\\Sample\\g";

def change():
    filenames = []
    for root, sub_folder, file_list in os.walk(fromPath):
        filenames += [os.path.join(root, file_path) for file_path in file_list]

    dic_kannji = {0: "あ", 1: "い", 2: "う", 3: "え", 4: "お", 5: "な", 6: "に", 7: "ぬ", 8: "ね", 9: "の",
                  10:"さ",11:"ざ",12:"き",13:"ぎ",14:"ほ",15:"ぼ",16:"ぽ",17:"は",18:"ば",19:"ぱ"}
    dic_code = {"あ":0, "い":1,  "う":2,  "え":3,  "お":4,  "な":5,  "に":6,  "ぬ":7,  "ね":8,  "の":9,
                  "さ":10,"ざ":11,"き":12,"ぎ":13,"ほ":14,"ぼ":15,"ぽ":16,"は":17,"ば":18,"ぱ":19}

    for file in filenames:
        im = Image.open(file)
        newname = file.replace('tif', 'png')

        for dic in dic_code.items():
            if dic[0] in newname:
                _newname = newname.replace(dic[0],str(dic[1]))
                im.save(_newname)  # or 'test.tif'
                continue;



change()