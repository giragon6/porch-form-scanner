from align_images import align_images
import numpy as np
import argparse
import imutils
import cv2

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required=True, help="Path to image to align to template image")
  ap.add_argument("-t", "--template", required=True, help="Path to template image")
  args = vars(ap.parse_args())

  img = cv2.imread(args["image"])
  tmp = cv2.imread(args["template"])
  aligned = align_images(img, tmp, debug=True)

  aligned = imutils.resize(aligned, width=700)
  template = imutils.resize(tmp, width=aligned.shape[1], height=aligned.shape[0])

  stacked = np.hstack([aligned, template])

  cv2.imshow("aligned", aligned)
  overlay = tmp.copy()
  overlay.resize((700,500))
  output = aligned.copy()
  output.resize((700,500))
  cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

  cv2.imshow("img alignment stacked", stacked)
  cv2.imshow("img alignment overlay", output)
  cv2.waitKey(0)
 
if __name__ == "__main__":
  main()