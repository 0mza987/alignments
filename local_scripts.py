# -*-coding:utf-8-*-

import os, json, cv2, re, time, signal
import zerorpc, base64, glob, itertools, traceback
import numpy as np

from multiprocessing import Pool
from Bio.pairwise2 import format_alignment
from Bio import pairwise2

PUNCT = [',', '.', '?', ':', '!', ';']


def main():
    
    LIST_name = ['5902d9e0f23a762911c2113e.png',
                 '0.5_a2f33914-118f-4223-a6f6-3ad071b8b370.jpg',
                 '1.0_be2ef15f-2d23-4b43-a848-0353a2c27a77.jpg',
                 '0.4_60de75ab-362d-4df2-8850-cdc4f93d9913.jpg']
    new_dict = {}
    align_data = json.load(open('./dataset/epoch2_30w.json'))
    for item in LIST_name:
        new_dict[item] = align_data[item]
    json.dump(new_dict, open('./dataset/badcase.json', 'w'))
        
        

if __name__=='__main__':
    main()




