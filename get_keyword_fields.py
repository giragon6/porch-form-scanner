import cv2
from paddleocr import PaddleOCR
import numpy as np
import os
from difflib import SequenceMatcher

def fuzzy_match(a, b, threshold=0.7):
    return SequenceMatcher(None, a, b).ratio() >= threshold

MODEL_DIR = './paddleocr_model'
os.makedirs(MODEL_DIR, exist_ok=True)
det_model_dir = os.path.join(MODEL_DIR, 'det')
rec_model_dir = os.path.join(MODEL_DIR, 'rec')
cls_model_dir = os.path.join(MODEL_DIR, 'cls')
textline_model_dir=os.path.join(MODEL_DIR, 'textline')
unwarp_model_dir=os.path.join(MODEL_DIR, 'unwarp')

ocr = PaddleOCR(use_textline_orientation=True, lang='en',
                text_detection_model_dir=det_model_dir,
                text_recognition_model_dir=rec_model_dir,
                doc_orientation_classify_model_dir=cls_model_dir,
                textline_orientation_model_dir=textline_model_dir,
                doc_unwarping_model_dir=unwarp_model_dir)

MARGIN = 600  # pixels to the right

"""
runs given image through OCR and attempts to detect adjacent (to the right) fields to provided OCR_LOCATIONS
Args:
    img: input image -> np.ndarray
    OCR_LOCATIONS: list of OCRLoc
Return:
    A dict containing the keywords and adjacent fields
    The image annotated with detected text bboxes
"""
def get_keyword_fields(input_img, OCR_LOCATIONS):
    result = ocr.predict(input_img)
    keyword_boxes = {}
    keyword_texts = {}
    annotated_img = input_img.copy()
    if isinstance(result[0], dict):
        res = result[0]
        polys = res['dt_polys'] if 'dt_polys' in res else res['rec_polys']
        scores = res['rec_scores']
        texts = res['rec_texts'] if 'rec_texts' in res else [''] * len(polys)
        margin_dict = {loc.text.lower(): getattr(loc, 'margin', MARGIN) for loc in OCR_LOCATIONS}
        for poly, score, text in zip(polys, scores, texts):
            pts = np.array(poly).astype(int)
            x1, y1 = pts[0]
            x2, y2 = pts[2]
            cv2.polylines(annotated_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            for loc in OCR_LOCATIONS:
                kw = loc.text.lower()
                if fuzzy_match(text.strip().lower(), kw, threshold=0.7):
                    keyword_boxes[kw] = (x1, y1, x2, y2)
                    keyword_texts[kw] = []
        for poly, score, text in zip(polys, scores, texts):
            pts = np.array(poly).astype(int)
            x1, y1 = pts[0]
            x2, y2 = pts[2]
            for kw, (kx1, ky1, kx2, ky2) in keyword_boxes.items():
                margin = margin_dict.get(kw, MARGIN)
                if x1 > kx2 and abs(y1 - ky1) < 120 and 0 < (x1 - kx2) < margin:
                    keyword_texts[kw].append((text, pts.tolist(), score))
    return keyword_texts, annotated_img