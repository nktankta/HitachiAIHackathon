from pyzbar.pyzbar import decode
import requests
from bs4 import BeautifulSoup
import cv2


def get_barcode_string(img):
    d = decode(img)
    try:
      code = d[0].data.decode("utf-8")
      return code
    except:
      return "error"

def clean_jancode(jancode: str):
    jancode = jancode.replace("-", "")
    return jancode

def get_item_name(jancode: str):
    jancode = clean_jancode(jancode)

    payload = {'jan': jancode}
    res = requests.post('https://www.janken.jp/goods/jk_catalog_syosai.php', params=payload)

    parsed_html = BeautifulSoup(res.content, "html.parser")

    item_name = parsed_html.find("h5", attrs={"id": "gname"}).get_text()
    seller_name = parsed_html.find_all(["td"], attrs={"class": "goodsval"})[2].get_text()
    item_name = ' '.join([seller_name, item_name])
    return item_name

def image2product(image):
    bar = get_barcode_string(image)
    clean_bar = clean_jancode(bar)
    if bar !="error":
        try:
            return get_item_name(clean_bar)
        except:
            "error"
    return "error"

if __name__ == '__main__':
    path = "barcode/P_20210616_121756.jpg"
    img = cv2.imread(path)
    print(image2product(img))