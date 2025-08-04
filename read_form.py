import torch
from align_images import align_images
from collections import namedtuple
import argparse
import imutils
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import numpy as np

from main import scan

def cleanup_text(text):
    return ''.join(c for c in text if ord(c) < 128).strip()

MODEL_DIR = "./trocr_model"
if os.path.exists(MODEL_DIR):
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
else:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to align to template image")
ap.add_argument("-t", "--template", required=True, help="Path to template image")
args = vars(ap.parse_args())

OCRLoc = namedtuple("OCRLoc", ["id", "bbox", "filter_keywords"])
OCR_LOCATIONS = [
    OCRLoc("beans", (620, 761, 195, 100), []),
    OCRLoc("sauce", (2076, 723, 241, 100), []),
]

unaligned_img = cv2.imread(args["image"])
if unaligned_img is None:
    raise ValueError("No image provided")

# from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
# scan_model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2, aux_loss=True) 
# scan_model.load_state_dict(torch.load('model_mbv3_iou_mix_2C049.pth', map_location='cpu'))
# scan_model.eval()

# img = scan(unaligned_img, scan_model)

tmp = cv2.imread(args["template"])
if tmp is None:
    raise ValueError("No template provided")

img_color = align_images(unaligned_img, tmp, debug=False)

img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

parsing_results = []

m = 50

for loc in OCR_LOCATIONS:
    x, y, w, h = loc.bbox
    box_template = tmp[y-m:y+h+m, x+m:x+w+m]
    box_input = img[y-m:y+h+m, x+m:x+w+m]
    sift = cv2.SIFT_create()
    kps_t, descs_t = sift.detectAndCompute(box_template, None)
    kps_i, descs_i = sift.detectAndCompute(box_input, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descs_i, descs_t, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    try:
        matched_vis = cv2.drawMatches(img, kps_i, tmp, kps_t, good_matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv2.imshow("matched keypoints", matched_vis)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
         
    if len(good_matches) >= 4:
        pts_i = np.float32([kps_i[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_t = np.float32([kps_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_i, pts_t, method=cv2.RANSAC)
        if H is not None:
            warped_box = cv2.warpPerspective(box_input, H, (w, h))
        else:
            print("not enough")
            warped_box = box_input[y:y+h, x:x+w]
    else:
        warped_box = box_input[y:y+h, x:x+w]
    pil_img = Image.fromarray(cv2.cvtColor(warped_box, cv2.COLOR_BGR2RGB))
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

results = {}
for loc, line in parsing_results:
    r = results.get(loc.id, None)
    if r is None:
        results[loc.id] = line, loc._asdict()
    else:
        existing_text, loc = r
        text = '{}\n{}'.format(existing_text, line)
        results[loc['id']] = text, loc

for loc_id, res in results.items():
    text, loc = res
    print(loc["id"])
    print("="*len(loc["id"]))
    print('{}\n\n'.format(text))
    x, y, w, h = loc["bbox"]
    clean = cleanup_text(text)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for i, line in enumerate(text.split('\n')):
        start_y = y + (i * 70) + 40
        cv2.putText(img, line, (x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

cv2.imshow("input", imutils.resize(img, width=700))
cv2.waitKey(0)