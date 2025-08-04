import cv2
import numpy as np
import imutils
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

from align_images import align_images

def find_and_extract_box(template_box, target_image, debug=False):
    # Convert to grayscale
    template_gray = cv2.cvtColor(template_box, cv2.COLOR_BGR2GRAY) if len(template_box.shape) == 3 else template_box
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if len(target_image.shape) == 3 else target_image
    # SIFT feature detection
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    if des1 is None or des2 is None:
        print("no descs found")
        return None, None
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    if debug:
        img_matches = cv2.drawMatches(template_box, kp1, target_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', imutils.resize(img_matches, width=1200))
        cv2.waitKey(0)
    if len(good_matches) < 4:
        print(f"Not enough matches found: {len(good_matches)}")
        return None, None
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Could not compute homography")
        return None, None
    h, w = template_box.shape[:2]
    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    target_corners = cv2.perspectiveTransform(template_corners, H)
    if debug:
        target_with_box = target_image.copy()
        cv2.polylines(target_with_box, [np.int32(target_corners)], True, (0, 255, 0), 3)
        cv2.imshow('Found Box', imutils.resize(target_with_box, width=800))
        cv2.waitKey(0)
    dst_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(target_corners.reshape(4, 2), dst_corners)
    extracted_box = cv2.warpPerspective(target_image, M, (w, h))
    return extracted_box, target_corners

# OCR setup
MODEL_DIR = "./trocr_model"
if os.path.exists(MODEL_DIR):
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
else:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image (warped form)")
    ap.add_argument("-t", "--template", required=True, help="Path to template image (reference form)")
    ap.add_argument("-b", "--box", nargs=4, type=int, required=True, help="Box coordinates in template: x y w h")
    ap.add_argument("--debug", action="store_true", help="Show debug visualizations")
    args = ap.parse_args()

    target_img = cv2.imread(args.image)
    template_img = cv2.imread(args.template)
    if target_img is None or template_img is None:
        print("Error loading images")
        return
    target_img = align_images(target_img, template_img, debug=True)
    x, y, w, h = args.box
    template_box = template_img[y:y+h, x:x+w]
    extracted_box, box_corners = find_and_extract_box(template_box, target_img, debug=args.debug)
    if extracted_box is not None:
        pil_img = Image.fromarray(cv2.cvtColor(extracted_box, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Recognized text:")
        print(text)
        cv2.imshow("Extracted Box", imutils.resize(extracted_box, width=400))
        cv2.waitKey(0)
    else:
        print("Box could not be found in the input image.")
    input("enter")

if __name__ == "__main__":
    main()
