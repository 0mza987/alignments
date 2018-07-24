import os, json, cv2, requests
import numpy as np


def get_data_from_api():
    URL = 'http://manhattanic.hexin.im/image/getList?'
    params = {
        '_limit': 100,
        '_page': 1
    }
    res = requests.get(URL, params=params).json()['data']
    json.dump(res, open('./dataset/kd_page_1.json', 'w'))

    for item in res:
        get_img(item['url'])


def get_img(url):
    splited_url = url.split('/')
    fpath = os.path.join('./dataset/kd_images', splited_url[-2])
    fname = splited_url[-1]
    if not os.path.exists(fpath): os.makedirs(fpath)
    
    img_url = 'http://192.168.1.115/{}/{}'.format(splited_url[-2], splited_url[-1])
    res = requests.get(img_url).content
    image = np.asarray(bytearray(res), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(fpath, fname), image)


if __name__ == '__main__':
    get_data_from_api()
    