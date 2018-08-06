# -*-coding:utf-8-*-
import os, json, cv2, re, time, signal
import zerorpc, base64, glob, itertools, traceback
import jellyfish
import numpy as np

from multiprocessing import Pool
from Bio.pairwise2 import format_alignment
from Bio import pairwise2

PUNCT = [',', '.', '?', ':', '!', ';']


def main():
    epoch_data = json.load(open('./dataset/epoch2.json'))
    for idx, item in enumerate(epoch_data.items()[0:100]):
        print 'Processing {} / {}'.format(idx+1, 1)
        fname, line_data = item
        truth = line_data['truth'].replace('$', ' ')
        pred = line_data['raw_text']
        print truth
        print pred
        try:
            alignments = pairwise2.align.globalmx(truth, pred, 2, -3)
            align1, align2, score, begin, end = alignments[-1]
            correct_str = format_alignment_index(align1, align2, score, begin, end, line_data)
        except:
            print traceback.format_exc()
        

def format_alignment_index(align1, align2, score, begin, end, line_data):
    """ use Bio.pairwise2.format_alignment as reference
        http://biopython.org/DIST/docs/api/Bio.pairwise2-pysrc.html#format_alignment
    """
    end = len(align2.strip('-'))
    end += 1 if (len(align1)>end and align1[end] in PUNCT) else 0
    print begin, end
    break_idx = []
    s = []
    s.append("%s\n" % align1)
    for a, b, str_idx in zip(align1[begin:end], align2[begin:end], range(begin, end)):
        if a == b: 
            s.append("|")  # match 
        elif a == "-" or b == "-": 
            s.append(" ")  # gap 
        else: 
            s.append(".")  # mismatch

        # 记录所有 match 的空格，例如 My name 中空格，用 | 表示
        if any([ a == b == ' ',
                 a == ' ' and b == '-',
                 a == '-' and b == ' '
            ]): break_idx.append(str_idx)

    s.append("\n") 
    s.append("%s\n" % align2)

    align1_copy = list(align1)
    align2_copy = list(align2)
    for idx in break_idx:
        align1_copy[idx] = '&'
        align2_copy[idx] = '&'
    align1_copy = ''.join(align1_copy)
    align2_copy = ''.join(align2_copy)
    
    print ''.join(s)
    return ''


if __name__=='__main__':
    main()




















