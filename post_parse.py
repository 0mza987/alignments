# -*-coding:utf-8-*-

import os, json, cv2, re, time, signal
import zerorpc, base64, glob, itertools, traceback
import numpy as np

from multiprocessing import Pool
from Bio.pairwise2 import format_alignment
from Bio import pairwise2

PUNCT = [',', '.', '?', ':', '!', ';']


def main():
    epoch_data = json.load(open('./dataset/epoch_res2_batch.json'))
    for idx, item in enumerate(epoch_data.items()[0:100]):
        print 'Processing {} / {}'.format(idx+1, len(epoch_data))
        fname, line_data = item
        truth = line_data['truth'].replace('$', ' ')
        truth = punct_clean(truth)
        pred = line_data['raw_text']
        pred = punct_clean(pred)
        print '@@@@@@@@@@@@@@@@@@@', truth
        print '@@@@@@@@@@@@@@@@@@@', pred 
        print line_data['pic_path']
        try:
            alignments = pairwise2.align.globalmx(truth, pred, 2, -3)
            align1, align2, score, begin, end = alignments[-1]
            correct_str = format_alignment_index(align1, align2, score, begin, end, line_data)
        except:
            print traceback.format_exc()

        
def punct_clean(sent):
    """ clean punctuation as OCR formant, such as "you ? -> you?"
    """
    for punct in PUNCT:
        sent = sent.replace('  %s' % punct, punct)
        sent = sent.replace(' %s' % punct, punct)
        sent = sent.replace(punct, '%s ' % punct)
        sent = sent.replace('%s   ' % punct, '%s ' % punct)
        sent = sent.replace('%s  ' % punct, '%s ' % punct)
    return sent.strip()


def format_alignment_index(align1, align2, score, begin, end, line_data):
    """ use Bio.pairwise2.format_alignment as reference
        http://biopython.org/DIST/docs/api/Bio.pairwise2-pysrc.html#format_alignment
    """

    # 文本长度以预测文本即align2的长度为准
    end = len(align2.strip('-'))
    end += 1 if (len(align1)>end and align1[end] in PUNCT) else 0
    align1 = align1[begin:end]
    align2 = align2[begin:end]
    # print begin, end
    break_idx = []

    # 生成如下格式的字符串显示
    #   st-udy. Most of students study English by class. We don't have
    #   |  |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #   s-iudy. Most of students study English by class. We don't have
    s = []
    s.append("%s\n" % align1)
    for a, b, str_idx in zip(align1, align2, range(begin, end)):
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

    # 将文本中本来存在的空格标记出来
    align1_copy = list(align1)
    align2_copy = list(align2)
    for idx in break_idx:
        align1_copy[idx] = '&'
        align2_copy[idx] = '&'
    align1_copy = ''.join(align1_copy)
    align2_copy = ''.join(align2_copy)
    
    
    # 开始配对
    correct_str = correction(align1_copy, align2_copy, line_data)
    return correct_str


def correction(align1, align2, line_data):
    # weights = line_data['prob']
    w_offset = 0
    correct_str = []
    for t, p in zip(align1.split('&'), align2.split('&')):
        t = t.replace('-', '')
        p = p.replace('-', '')
        if t==p: 
            correct_str.append(t)
        elif line_data[t] > line_data[p]:
            correct_str.append(t)
        else:
            correct_str.append(p)
    return ' '.join(correct_str)


if __name__=='__main__':
    main()




















