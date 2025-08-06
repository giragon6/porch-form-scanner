import gradio as gr
import numpy as np
from PIL import Image
import torch
import cv2
from get_keyword_fields import get_keyword_fields
import gc

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torchvision.transforms as T

def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

preprocess_transforms = image_preprocess_transforms()
model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2, aux_loss=True)
model.load_state_dict(torch.load('model_mbv3_iou_mix_2C049.pth', map_location='cpu'))
model.eval()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()
  
def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

def scan(image_true, trained_model, image_size=384):
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2
    imH, imW, C = image_true.shape
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)
    with torch.no_grad():
        out = trained_model(image_model)["out"]
    del image_model
    gc.collect()
    out = torch.argmax(out, dim=1, keepdim=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape
    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()
    del _out_extended
    gc.collect()
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)
    corners[:, 0] -= half
    corners[:, 1] -= half
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y
    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):
        pass  # TODO: handle this
    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    src = np.array(corners, dtype="float32")
    dst = np.array(destination_corners, dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    maxWidth = int(max(dst[:,0]))
    maxHeight = int(max(dst[:,1]))
    warped = cv2.warpPerspective(image_true, M, (maxWidth, maxHeight))
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def ocr_pipeline(input_image, ocr_fields):
    img = np.array(input_image.convert('RGB'))
    cropped = scan(img, model)
    OCR_LOCATIONS = []
    for line in ocr_fields.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 1 and parts[0]:
            text = parts[0]
            id_val = text.lower().replace(' ', '_')
            margin = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 600
            OCR_LOCATIONS.append(type('OCRLoc', (), {})(id=id_val, text=text, margin=margin))
    keyword_fields, annotated_img = get_keyword_fields(cropped, OCR_LOCATIONS)
    table = ""
    for kw, items in keyword_fields.items():
        for t, box, score in items:
            table += f"{kw}: {t} (score: {score:.2f})\n"
    annotated_pil = Image.fromarray(annotated_img.astype(np.uint8))
    return annotated_pil, table

demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=[
        gr.Image(type="pil", label="Upload Document Image"),
        gr.Textbox(lines=4, label="OCR Fields (one per line, format: text[,margin(px)])", value="Beans\nSauce/Canned Tomatoes\nPeanut Butter,100\nOther\nPicking up for")
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Extracted Fields")
    ],
    title="Document Scanner Demo",
    description="Upload a document image and specify fields to extract."
)

if __name__ == '__main__':
    demo.launch()