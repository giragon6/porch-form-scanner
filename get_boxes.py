import argparse
import os
import re
import cv2
import pytesseract
from collections import namedtuple
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from align_images import align_images

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to align to template image")
ap.add_argument("-t", "--template", required=True, help="Path to template image")
args = vars(ap.parse_args())

OCRLoc = namedtuple("OCRLoc", ["id", "text", "filter_keywords"])
OCR_LOCATIONS = [
    OCRLoc("beans", "Beans", []),
    OCRLoc("sauce", "Sauce/Canned Tomatoes", []),
]

unaligned_img = cv2.imread(args["image"])
if unaligned_img is None:
    raise ValueError("No image provided")

template_img = cv2.imread(args["template"])
if template_img is None:
    raise ValueError("No template provided")

input_img = align_images(unaligned_img, template_img, debug=True)

ocr_data = pytesseract.image_to_data(input_img, output_type=pytesseract.Output.DICT)
print(ocr_data)

keyword_boxes = {}
for loc in OCR_LOCATIONS:
    keyword = loc.text.lower()
    for i, word in enumerate(ocr_data['text']):
        if word.strip().lower() == keyword:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            keyword_boxes[keyword] = (x, y, w, h)
            break
print(keyword_boxes)

MODEL_DIR = "./trocr_model"
if os.path.exists(MODEL_DIR):
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
else:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

parsing_results = []

for loc in OCR_LOCATIONS:
    if loc.id not in keyword_boxes:
        continue
    x, y, w, h = keyword_boxes[loc.id]
    field_crop = input_img[y-50:y+h+50, x:x+w+600]
    cv2.imshow(f"Field: {loc.id}", field_crop)
    pil_img = Image.fromarray(field_crop)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    for line in text.split("\n"):
        if len(line) == 0:
            continue
        lower = line.lower()
        count = sum([lower.count(x) for x in loc.filter_keywords])
        if count == 0:
            parsing_results.append((loc, line))

NUM_REG = r'\d+'

for loc, txt in parsing_results:
    txt_nums = re.findall(NUM_REG, txt)
    print(f"{loc.id}: {txt.strip()} ({''.join(txt_nums)})")

input('press ENTER to exit')
cv2.destroyAllWindows()