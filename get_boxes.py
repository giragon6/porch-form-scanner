import argparse
import cv2
import imutils
from paddleocr import PaddleOCR
from collections import namedtuple
import numpy as np
import os

from align_images import align_images

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to align to template image")
ap.add_argument("-t", "--template", required=True, help="Path to template image")
args = vars(ap.parse_args())

def fuzzy_match(a, b, threshold=0.7):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= threshold

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

input_img = align_images(unaligned_img, template_img, debug=False)

MODEL_DIR = './paddleocr_model'
os.makedirs(MODEL_DIR, exist_ok=True)
det_model_dir = os.path.join(MODEL_DIR, 'det')
rec_model_dir = os.path.join(MODEL_DIR, 'rec')
cls_model_dir = os.path.join(MODEL_DIR, 'cls')

ocr = PaddleOCR(use_textline_orientation=True, lang='en',
                text_detection_model_dir=det_model_dir,
                text_recognition_model_dir=rec_model_dir)

result = ocr.predict(input_img)

MARGIN = 600  # pixels to the right

keyword_boxes = {}
keyword_texts = {}
if isinstance(result[0], dict):
    res = result[0]
    polys = res['rec_polys']
    scores = res['rec_scores']
    texts = res['rec_texts'] if 'rec_texts' in res else [''] * len(polys)
    img_vis = input_img.copy()
    for loc in OCR_LOCATIONS:
        kw = loc.text.lower()
        for poly, score, text in zip(polys, scores, texts):
            pts = np.array(poly).astype(int)
            x1, y1 = pts[0]
            x2, y2 = pts[2]
            cv2.polylines(img_vis, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(img_vis, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            print(f'Text: {text}, Box: {pts.tolist()}, Score: {score}')
            if fuzzy_match(text.strip().lower(), kw, threshold=0.7):
                keyword_boxes[kw] = (x1, y1, x2, y2)
                keyword_texts[kw] = []

    print(keyword_texts)
    for poly, score, text in zip(polys, scores, texts):
        pts = np.array(poly).astype(int)
        x1, y1 = pts[0]
        x2, y2 = pts[2]
        for kw, (kx1, ky1, kx2, ky2) in keyword_boxes.items():
            if x1 > kx2 and abs(y1 - ky1) < 100 and 0 < (x1 - kx2) < MARGIN:
                keyword_texts[kw].append((text, pts.tolist(), score))
    img_vis = imutils.resize(img_vis, width=700)
    cv2.imshow("PaddleOCR Results", img_vis)
    input("press ENTER to continue")
    for kw, items in keyword_texts.items():
        print(f"Keyword: {kw}")
        for t, box, score in items:
            print(f"  Associated text: {t}, Box: {box}, Score: {score}")
else:
    print("Unexpected result format from PaddleOCR:", type(result), result)

# MODEL_DIR = "./trocr_model"
# if os.path.exists(MODEL_DIR):
#     processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
#     model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
# else:
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     processor.save_pretrained(MODEL_DIR)
#     model.save_pretrained(MODEL_DIR)

parsing_results = []

for loc in OCR_LOCATIONS:
    if loc.id not in keyword_boxes:
        continue
    x1, y1, x2, y2 = keyword_boxes[loc.id]
    field_crop = input_img[y1:y2, x1:x2]
    cv2.imshow(f"Field: {loc.id}", field_crop)
    input("press ENTER to continue")
    # pil_img = Image.fromarray(field_crop)
    # pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    # generated_ids = model.generate(pixel_values)
    # text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    # for line in text.split("\n"):
    #     if len(line) == 0:
    #         continue
    #     lower = line.lower()
    #     count = sum([lower.count(x) for x in loc.filter_keywords])
    #     if count == 0:
    #         parsing_results.append((loc, line))

# NUM_REG = r'\d+'

# for loc, txt in parsing_results:
#     txt_nums = re.findall(NUM_REG, txt)
#     print(f"{loc.id}: {txt.strip()} ({''.join(txt_nums)})")

input('press ENTER to exit')
cv2.destroyAllWindows()