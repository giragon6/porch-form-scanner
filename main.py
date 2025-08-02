from flask import Flask, flash, redirect, request, render_template_string, url_for
import torch
import numpy as np
import cv2
import io
from PIL import Image
import torchvision.transforms as T
import gc
from matplotlib import pyplot as plt
import pytesseract
import base64

def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

preprocess_transforms = image_preprocess_transforms()

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
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

def scan(image_true, trained_model, image_size=384, BUFFER=10):
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

def get_text(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_gray, cmap='gray')
    ax.set_title("Image with Text Bounding Boxes")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    txt = pytesseract.image_to_string(img_gray)
    return txt, plot_data

app = Flask(__name__)
app.secret_key = 'secret'

HTML = '''
<!doctype html>
<title>Document Scanner Flask</title>
<h1>Document Scanner: Semantic Segmentation using DeepLabV3-MobilenetV3</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value=Upload>
</form>
{% if result_url %}
  <h2>Scanned Result:</h2>
  <img src="{{ result_url }}" style="max-width: 500px;">
  <br><a href="{{ result_url }}" download>Download Result</a>
{% endif %}
{% if plot_data %}
  <h2>Text Detection Plot:</h2>
  <img src="data:image/png;base64,{{ plot_data }}" style="max-width: 500px;">
{% endif %}
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>{% for message in messages %}<li>{{ message }}</li>{% endfor %}</ul>
  {% endif %}
{% endwith %}
'''
    
import os

@app.route('/', methods=['GET', 'POST'])
def index():
    result_url = None
    plot_data = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        try:
            img = Image.open(file.stream).convert('RGB')
            img_np = np.array(img)
            cropped = scan(img_np, model)
            text, plot_data = get_text(cropped)
            flash(text)   
            cropped_img = Image.fromarray(cropped.astype(np.uint8))
            buf = io.BytesIO()
            cropped_img.save(buf, format='PNG')
            buf.seek(0)
            temp_path = 'static/result.png'
            os.makedirs('static', exist_ok=True)
            cropped_img.save(temp_path)
            result_url = url_for('static', filename='result.png')
        except Exception as e:
            flash(f'Processing failed: {e}')
    return render_template_string(HTML, result_url=result_url, plot_data=plot_data)
   
if __name__ == '__main__':
    app.run(debug=True)     