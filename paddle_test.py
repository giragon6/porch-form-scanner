from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = 'aligned-form.png'
img = cv2.imread(img_path)

result = ocr.predict(img)

if isinstance(result[0], dict):
    res = result[0]
    polys = res['rec_polys']
    scores = res['rec_scores']
    texts = res['rec_texts'] if 'rec_texts' in res else [''] * len(polys)
    img_vis = img.copy()
    for poly, score, text in zip(polys, scores, texts):
        pts = np.array(poly).astype(int)
        cv2.polylines(img_vis, [pts], isClosed=True, color=(0,255,0), thickness=2)
        x, y = pts[0]
        cv2.putText(img_vis, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        print(f'Text: {text}, Box: {pts.tolist()}, Score: {score}')
    cv2.imshow("PaddleOCR Results", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Unexpected result format from PaddleOCR:", type(result), result)
