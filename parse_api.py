# -*-coding:utf-8-*-
# -------------------------------------
# ocr text alignment
# -------------------------------------

import os, json, cv2, re, jellyfish
import zerorpc, base64, glob, itertools
import numpy as np

if os.name == 'nt':
    d = [k.strip() for k in open('dict/american-english', 'r').readlines()]
    d.extend([k.capitalize().strip() for k in open('dict/american-english', 'r').readlines()])
else:
    import enchant
    d = enchant.Dict("en_US")
    dict_word = dict()
    for line in open('dict/word_freq.txt', 'r').readlines():
        line = line.strip()
        word = line.split('\t')[0]
        freq = int(line.split('\t')[1])
        if freq >= 10 and d.check(word):
            dict_word[word] = freq


LIST_not_in_k12 = ['frost', 'torn', 'pus']
from Bio.pairwise2 import format_alignment
from Bio import pairwise2


def _img_to_str_base64(image):
    """ convert image to base64 string 
    """
    img_encode = cv2.imencode('.jpg', image)[1]    
    img_base64 = base64.b64encode(img_encode)
    return img_base64


def punt_clean(sent):
    """ clean punctuation as OCR formant, such as you? -> you ?
    """
    for punt in ['.', '?', '!']:
        sent = sent.replace(punt, ' %s ' % punt).replace('  %s  ' % punt, ' %s ' % punt).replace('  %s' % punt, ' %s' % punt).replace('%s  ' % punt, '%s ' % punt)
    return sent.strip()


def html_clean(sent):
    LIST_html = [('&lt;', '<'), ('&lt;', '>'), ('&#39;', "'")]
    for item in LIST_html:
        sent = sent.replace(item[0], item[1])

    for punt in ['.', '?', '!', ',']:
        sent = sent.replace(punt, ' %s ' % punt).replace('  %s  ' % punt, ' %s ' % punt).replace('  %s' % punt, ' %s' % punt).replace('%s  ' % punt, '%s ' % punt)
    return sent


def get_text_size(text):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    margin = 1
    thickness = 1
    color = (255, 255, 255)
    size = cv2.getTextSize(text, font, font_scale, thickness)

    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin
    return text_width, line_height


def IoU(Reframe, GTframe):
    """ 自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）坐标。
    """
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2] - Reframe[0];
    height1 = Reframe[3] - Reframe[1];

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2] - GTframe[0];
    height2 = GTframe[3] - GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height; # 两矩形相交面积
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    # return IOU

    # 更新判断指标，矩形的长、宽应该接近和类似
    FLAG_width  = abs(width1 - width2) <= 0.15 * max(width1, width2)
    FLAG_height = abs(height1 - height2) <= 0.1 * max(height1, height2)
    return ratio, FLAG_width and FLAG_height



def format_alignment_index(align1, align2, score, begin, end, line_data):
    """ use Bio.pairwise2.format_alignment as reference
        http://biopython.org/DIST/docs/api/Bio.pairwise2-pysrc.html#format_alignment
    """
    break_idx = list()

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

    # My  name  is  Tang  Lo - I  was  born  in  a  small  town  in  thejung
    # ||| |.||| ||| |||| ||.| || ||||| ||||| ||| || |||||| ||.| ||| |.|||.||
    # My -nome -is -Tang- Li . I- was -born -in -a -small -torn- in- Zhejing
    # break_idx = [2, 8, 12, 19, 22, 24, 27, 31, 37, 41, 44, 51, 58, 62]

    align1_copy = list(align1)
    align2_copy = list(align2)
    for idx in break_idx:
        align1_copy[idx] = '#'
        align2_copy[idx] = '#'
    align1_copy = ''.join(align1_copy)
    align2_copy = ''.join(align2_copy)

    # 单词匹配与矫正算法
    # 1. 如 a == b，正确通过
    # 2. 如 a, b 有一方为标点，另一方为 -，取标点
    # 3. 如 a, b 有一方为单词，且 OCR 识别概率 <= 0.9，取单词部分
    # 4. 如 a, b 中出现 |，则直接添加删除
    # 5. 如 a, b 都是单词，如 b <= 0.75，则取 a 结果作为单词

    LIST_punts = ['.', ',', '?', '!']
    correct_sent = list()
    word_idx = 0
    for a, b in zip(align1_copy.split('#'), align2_copy.split('#')):

        if a == b == '': continue
        if a == b:
            correct_sent.append(a)
            if a == b == '-': word_idx -= 1     # 如两者相等，则去除 - 对 word_idx 影响（我们自己 API 容易漏 -）
        elif a == '-' and b in LIST_punts:
            correct_sent.append(b)
        elif a in LIST_punts and b == '-':
            correct_sent.append(a)
        elif '|' in a or '|' in b:
            if '|' in b[0 : int(0.5 * len(b))]:
                correct_sent.append('|')
                correct_sent.append(b.replace('|', '').replace('-', ''))
            else:
                correct_sent.append(b.replace('|', '').replace('-', ''))
                correct_sent.append('|')

        # 判断 a, b 是否为正确单词
        else:
            if '-' in a: a = a.replace('-', '')
            if '-' in b: b = b.replace('-', '')

            # of  books  and --- DVDS  about  maths - I  loved  them
            # ||| |||||| ||||   |      |||||| |||||| ||| |||||| ||||
            # of -books -and pus ------about -maths . I -loved -them
            # 当 b 为 -, a 不为 - 时，做处理
            if b == '' and a != '':
                correct_sent.append(a)

            if os.name == 'nt':
                a_check = True if a == '' else all([a != '', a not in LIST_not_in_k12, a in d])
                b_check = True if b == '' else all([b != '', b not in LIST_not_in_k12, b in d])
            else:
                a_check = True if a == '' else all([a != '', a not in LIST_not_in_k12, d.check(a)])
                b_check = True if b == '' else all([b != '', b not in LIST_not_in_k12, d.check(b)])

            # print line_data
            # print word_idx, line_data[word_idx]
            # 可能有溢出情况发生
            if word_idx < len(line_data):
                ocr_prob = line_data[word_idx]['weight']
                
                if ocr_prob >= 0.95 and line_data[word_idx]['word'] != 'punt':
                    correct_sent.append(b)  # 如果 ocr prob 绝对高，直接过滤所有正确、错误情况
                elif a_check == b_check:
                    prov_val = 0.9 if any([a.isdigit(), b.isdigit()]) else 0.8
                    if ocr_prob >= prov_val: correct_sent.append(b)
                    else: correct_sent.append(a)
                elif b_check == True and a_check == False:
                    correct_sent.append(b)
                elif a_check == True and b_check == False:
                    correct_sent.append(a)

            # print a, a_check, b, b_check, line_data[word_idx]['word'], line_data[word_idx]['weight']
            

        if word_idx < len(line_data):
            word_idx += 1
    
    correct_str = ' '.join(correct_sent).replace('  ', ' ')
    print '**********************************'
    print correct_str, '\t', 'Align score:', score
    print '**********************************'
    print ''.join(s)
    print '**********************************'
    return correct_str


def show_result_on_image(result, image):
    """ Display the obtained results onto the input image
    """
    LIST_word_ims = list()
    image_copy = image.copy()
    lines = result['recognitionResult']['lines']
    for i in range(len(lines)):
        # print lines[i]['text'].replace('&#39;', "'")
        words = lines[i]['words']
        for j in range(len(words)):
            tl = (words[j]['boundingBox'][0], words[j]['boundingBox'][1])
            tr = (words[j]['boundingBox'][2], words[j]['boundingBox'][3])
            br = (words[j]['boundingBox'][4], words[j]['boundingBox'][5])
            bl = (words[j]['boundingBox'][6], words[j]['boundingBox'][7])
            text = words[j]['text']
            text = text.replace('&#39;', "'")
            if text.strip() == '': continue
            
            x = [tl[0], tr[0], tr[0], br[0], br[0], bl[0], bl[0], tl[0]]
            y = [tl[1], tr[1], tr[1], br[1], br[1], bl[1], bl[1], tl[1]]

            x0 = min(x)
            x1 = max(x)
            y0 = min(y)
            y1 = max(y)

            save_name = str(j) + '_' + text + '.jpg'
            image_text = image_copy[y0: y1, x0 : x1]
            LIST_word_ims.append((image_text, save_name))
            # cv2.imwrite(save_name, image_text)
            # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            # cv2.putText(image, text, (int(0.5 * (x0 + x1)), y0), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    return image, LIST_word_ims



def main():
    c_en_predict = zerorpc.Client(heartbeat=None, timeout=20)
    c_en_predict.connect('tcp://192.168.1.115:12001')  # en 12001, cn 11001

    # 1. 将所有图像进行 RPC ocr 识别，得到识别结果
    # 将 sample/*.jpg 替换为给你的 Folders ---

    LIST_test = glob.glob(r'./sample/*.jpg')
    LIST_test.sort()
    for idx, FILE_image in enumerate(LIST_test[0:]):

        print FILE_image 
        image_vis = cv2.imread(FILE_image)
        image     = cv2.imread(FILE_image, 0)

        # -------------------------------------
        # OCR RPC result
        # -------------------------------------
        # 本地 RPC ocr result 地址
        FILE_rpc_ocr = 'rpc_res/' + os.path.basename(FILE_image) + '.rpc.res.json'
        FILE_new_api_ocr = FILE_image + '.ocr.json'
        if os.path.exists(FILE_rpc_ocr) == False:
            LIST_data = {'fname': os.path.basename(FILE_image), 'img_str': _img_to_str_base64(image)}
            rpc_ocr_res = c_en_predict.predict_essay(LIST_data, True, [])
            with open(FILE_rpc_ocr, 'w') as o: print >> o, json.dumps(rpc_ocr_res)

        # -------------------------------------
        # load 2 OCR result
        # -------------------------------------
        print FILE_rpc_ocr
        DATA_rpc_json = json.load(open(FILE_rpc_ocr))                          # rpc.ocr.json
        DATA_new_api_json = json.load(open(FILE_new_api_ocr))                      # ocr.json

        # -------------------------------------
        # init for msapi result
        # -------------------------------------
        # 考虑到标点符号占位问题，将标点作为一个 dict_line 对象加入进去
        dict_line = dict()
        rpc_ocr_result = json.loads(DATA_rpc_json['data'])['blocks'][0]
        for inst in rpc_ocr_result['words']['words']:
            line_idx = inst['line']
            if dict_line.has_key(line_idx) == False: dict_line[line_idx] = list()
            dict_line[line_idx].append(inst)

            # 如单词字符内（如，10th.）有标点符号，则继续 +1
            if inst['word'] not in ['.', ',', '!', '?']:
                count_punt = sum([inst['word'].count(p) for p in ['.', ',', '!', '?']])
                for _ in range(0, count_punt):
                    dict_line[line_idx].append({'word': 'punt', 'weight': 1.0})

        # -------------------------------------
        # mapping msapi to rpcapi
        # -------------------------------------
        # 将结果映射到同一文本行内，方便后续 alignment

        # Think: 可否用 kmeans 聚类方式来搞？
        rpc_ocr_lines = len([line for line in rpc_ocr_result['lines'] if len(line['text'].strip()) > 0])
        print 'rpc api get %s lines' % rpc_ocr_lines

        LIST_line_feature = list()
        for line_idx, line_ocr in enumerate(rpc_ocr_result['lines']):       # 不考虑 API 识别结果
            y0_root = line_ocr['top']
            y1_root = line_ocr['bottom']
            # LIST_line_feature.append((y0_root, y1_root, y1_root - y0_root))

        new_api_line_count = len(DATA_new_api_json['recognitionResult']['lines'])
        print 'new api get %s lines' % new_api_line_count
        for idx, line_api in enumerate(DATA_new_api_json['recognitionResult']['lines']):
            y0 = min(line_api['boundingBox'][1], line_api['boundingBox'][3], line_api['boundingBox'][5], line_api['boundingBox'][7])
            y1 = max(line_api['boundingBox'][1], line_api['boundingBox'][3], line_api['boundingBox'][5], line_api['boundingBox'][7])
            LIST_line_feature.append((y0, y1, y1 - y0))

        # 进行 kmeans?
        from sklearn.cluster import KMeans
        X = np.array(list(LIST_line_feature))
        kmeans = KMeans(n_clusters=rpc_ocr_lines, random_state=0).fit(X)
        print sorted(kmeans.labels_)
        kmeans_category_count = len(set(kmeans.labels_))
        print 'Kmeans labels category count:', kmeans_category_count
        # 将 cluster label 进行聚类
        LIST_lines = list()
        for k in range(0, rpc_ocr_lines):
            cluster_lines = [DATA_new_api_json['recognitionResult']['lines'][i] for i, x in enumerate(kmeans.labels_) if x == k]
            clutser_y = [(i['boundingBox'][1], i['boundingBox'][3], i['boundingBox'][5], i['boundingBox'][7]) for i in cluster_lines]
            y0 = min([min(i) for i in clutser_y])
            y1 = max([max(i) for i in clutser_y])
            print k, html_clean(' '.join([i['text'] for i in cluster_lines])), y0, y1
            LIST_lines.append((y0, y1, cluster_lines))

        for line_idx, line_ocr in enumerate(rpc_ocr_result['lines']):
            line_idx += 1 # 代表 line，从 1 开始
            y0_root = line_ocr['top']
            y1_root = line_ocr['bottom']
            line_height = abs(y1_root - y0_root)
            area_seams = [0, y0_root, 100, y1_root]

            print '-' * 50
            print 'OCR_text:',line_ocr['text']
            LIST_line = list()
            for idx, inst in enumerate(LIST_lines):
                y0, y1, line_api = inst

                line_text = html_clean(' '.join([i['text'] for i in line_api]))
                str_similar = jellyfish.jaro_distance(line_ocr['text'], line_text)
                area_api = [0, y0, 100, y1]
                hit_ratio, _ = IoU(area_seams, area_api)
                height_diff = abs(abs(y1_root - y0_root) - abs(y1 - y0))

                if any([str_similar >= 0.75, hit_ratio >= 0.5]):
                    print 'API_text:',line_text, str_similar, hit_ratio
                    ocr_text = html_clean(line_ocr['text'])
                    api_text = html_clean(line_text)

                    alignments = pairwise2.align.globalms(api_text, ocr_text, 2, -1, -2, -0.1)
                    if '' in [api_text.strip(), ocr_text.strip()]: continue
                    # for item in alignments:
                    #     print item[0]
                    #     print item[1]
                    #     print item[2:]
                    align1, align2, score, begin, end = alignments[-1]
                    print '~~~~~~~~~~~~~~'
                    print 'API_text:',api_text
                    print 'OCR_text:',ocr_text
                    print '~~~~~~~~~~~~~~'
                    print line_idx
                    correct_sent = format_alignment_index(align1, align2, score, begin, end, dict_line[line_idx])

                    text_width, line_height = get_text_size(correct_sent)
                    cv2.rectangle(image_vis, (10, y0 - 10), (10 + text_width, y0 - 10 + line_height), (255, 180, 0), cv2.FILLED)
                    cv2.putText(image_vis, correct_sent, (10, y0 + 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite('./k-%s.jpg' % os.path.basename(FILE_image), image_vis)

if __name__ == '__main__':
    main()