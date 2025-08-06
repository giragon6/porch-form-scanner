import numpy as np
import cv2
import imutils
import torch
from main import scan

def align_images(img_unaligned, template, max_features=2000, keep_percent=0.5, ratio_thresh=0.75, debug=False):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2, aux_loss=True)
    model.load_state_dict(torch.load('model_mbv3_iou_mix_2C049.pth', map_location='cpu'))
    model.eval()

    image = scan(img_unaligned, model)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(max_features)
    kps_i, descs_i = sift.detectAndCompute(img_gray, None)
    kps_t, descs_t = sift.detectAndCompute(tmp_gray, None)

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descs_i, descs_t, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"Number of good matches: {len(good_matches)}")
    keep = int(len(good_matches) * keep_percent)
    matches = sorted(good_matches, key=lambda x: x.distance)[:keep]

    if debug:
        matched_vis = cv2.drawMatches(image, kps_i, template, kps_t, matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv2.imshow("matched keypoints", matched_vis)
        cv2.waitKey(0)

    if len(matches) < 4:
        raise ValueError("Not enough good matches to compute homography.")

    pts_i = np.zeros((len(matches), 2), dtype="float32")
    pts_t = np.zeros((len(matches), 2), dtype="float32")
    for i, m in enumerate(matches):
        pts_i[i] = kps_i[m.queryIdx].pt
        pts_t[i] = kps_t[m.trainIdx].pt

    H, mask = cv2.findHomography(pts_i, pts_t, method=cv2.RANSAC)
    if H is None:
        raise ValueError("Homography could not be computed.")
    
    h, w = template.shape[:2]

    # aligned = cv2.warpPerspective(image, H, (w, h))
    aligned = cv2.resize(image, (w,h))

    return aligned