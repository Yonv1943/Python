import re
import os
import time
import urllib.request

import cv2
import numpy as np
from bs4 import BeautifulSoup


def url_to_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    url = urllib.request.Request(url=url, headers=headers)

    response = urllib.request.urlopen(url)
    img = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def url_to_text(url, tag, tag_class):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    url = urllib.request.Request(url=url, headers=headers)

    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response)
    text = soup.find_all(tag, tag_class)
    return text


def collect_image_from_url(url, tag, tag_class, patten):
    url_text = url_to_text(url, tag, tag_class)
    print("| get url_text")
    # print("url_text:", url_text)

    image_id_s = []
    for item in url_text:
        findall_result = re.findall(patten, str(item))
        image_id_s.extend(findall_result)
    print("| get image_id_s")
    # [print(item) for item in image_id_s]

    url_img = url[:url.index('/', url.index('//') + 2)]
    img_save_dirt = url_img.split('/')[-1]
    os.mkdir(img_save_dirt) if not os.path.exists(img_save_dirt) else None
    for image_id in image_id_s:
        img = url_to_image(url=url_img + image_id)
        img_save_name = image_id.split('/')[-1]
        img_save_path = os.path.join(img_save_dirt, img_save_name)

        cv2.imwrite(img_save_path, img), print(img_save_name)
        cv2.imshow('', img), cv2.waitKey(1)

    # for img in [f for f in os.listdir(os.getcwd())if f[-4:] == '.jpg']:
    #     img = cv2.imread(img)
    #     cv2.imshow('', img), cv2.waitKey(234)


def collect_images_from_urls(url_img_s):
    img_save_dirt = "weather.gc.ca.jpg_dir"
    os.mkdir(img_save_dirt) if not os.path.exists(img_save_dirt) else None
    try:
        for url_img in url_img_s:
            img_save_name = "%s_%s.jpg" % (url_img.split('/')[-1], int(time.time()))
            img_save_path = os.path.join(img_save_dirt, img_save_name)
            img = url_to_image(url_img)
            cv2.imwrite(img_save_path, img), print(img_save_name)
            # cv2.imshow('', img), cv2.waitKey()
            time.sleep(0.1943)
    except Exception as e:
        print("|Error:", e)


def run():
    # url = "https://weather.gc.ca/satellite/index_e.html"
    # patten = re.compile(r"src=\"(.*.jpg)\"")
    # tag, tag_class = 'img', None
    # collect_image_from_url(url, tag, tag_class, patten)

    url = "http://www.bom.gov.au/australia/satellite/"
    patten = re.compile(r"url: \"(.*\.jpg)\"")
    tag, tag_class = 'script', None

    url_img_s = [
        "https://weather.gc.ca/data/satellite/goes_nam_visiblex_100.jpg",
        "https://weather.gc.ca/data/satellite/goes_nam_1070x_100.jpg",
        "https://weather.gc.ca/data/satellite/goes_nam_1070_100.jpg",
    ]

    time1 = time2 = 0
    while True:
        mod_time1 = int(time.time() % 86400)  # 60*60*24=86400
        if mod_time1 > time1:
            time1 = mod_time1
            collect_image_from_url(url, tag, tag_class, patten)
            print('|')

        mod_time2 = int(time.time() % 900)  # 60*15
        if mod_time2 > time2:
            collect_images_from_urls(url_img_s)
            print('|')

        time.sleep(178)


if __name__ == '__main__':
    run()

"""
Reference:
http://python.jobbole.com/81131/
https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
"""
