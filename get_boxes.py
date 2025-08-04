import argparse
import cv2
import pytesseract
from collections import namedtuple
import numpy as np
from PIL import Image

from align_images import align_images

OCRLoc = namedtuple("OCRLoc", ["id", "bbox", "filter_keywords"])
OCR_LOCATIONS = [
    OCRLoc("beans", (197, 759, 908, 100), []),
    OCRLoc("sauce", (1564, 728, 900, 100), []),
]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to align to template image")
ap.add_argument("-t", "--template", required=True, help="Path to template image")
args = vars(ap.parse_args())

OCRLoc = namedtuple("OCRLoc", ["id", "bbox", "filter_keywords"])
OCR_LOCATIONS = [
    OCRLoc("beans", (197, 649, 908, 120), []),
    OCRLoc("sauce", (1564, 628, 900, 100), []),
]

unaligned_img = cv2.imread(args["image"])
if unaligned_img is None:
    raise ValueError("No image provided")

template_img = cv2.imread(args["template"])
if template_img is None:
    raise ValueError("No template provided")

input_img = align_images(unaligned_img, template_img, debug=False)

ocr_data = pytesseract.image_to_data(template_img, output_type=pytesseract.Output.DICT)

keyword_boxes = {}
for loc in OCR_LOCATIONS:
    keyword = loc.id.lower()
    for i, word in enumerate(ocr_data['text']):
        if word.strip().lower() == keyword:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            keyword_boxes[keyword] = (x, y, w, h)
            break
  
for loc in OCR_LOCATIONS:
    x, y, w, h = loc.bbox
    field_crop = input_img[y:y+h, x:x+w]
    cv2.imshow(f"Field: {loc.id}", field_crop)
    pil_img = Image.fromarray(field_crop)
    field_text = pytesseract.image_to_string(pil_img)
    print(f"{loc.id}: {field_text.strip()}")

cv2.waitKey(0)
cv2.destroyAllWindows()