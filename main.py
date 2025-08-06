from collections import namedtuple
from flask import Flask, redirect, request, render_template_string, url_for, session, Response
import torch
import numpy as np
import cv2
import io
from PIL import Image
import torchvision.transforms as T
import gc
from get_keyword_fields import get_keyword_fields
import csv

OCRLoc = namedtuple("OCRLoc", ["id", "text", "margin"])

DEFAULT_OCR_LOCATIONS = [
    OCRLoc("beans", "Beans", 600),
    OCRLoc("sauce_canned_tomatoes", "Sauce/Canned Tomatoes", 600),
    OCRLoc("peanut_butter", "Peanut Butter", 200),
    OCRLoc("other", "Other", 1000),
    OCRLoc("picking_up_for", "Picking up for", 600)
]

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

app = Flask(__name__)
app.secret_key = 'secret'

HTML = '''
<!doctype html>
<title>Document Scanner</title>
<h1>Document Scanner Demo</h1>
<h3>Automatically extract data from your documents with no prior setup</h3>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul id="flash-messages">{% for message in messages %}<li>{{ message }}</li>{% endfor %}</ul>
  {% endif %}
{% endwith %}
<form method=post enctype=multipart/form-data id="upload-form">
  <input type=radio name=img_mode value="demo" {% if img_mode == 'demo' %}checked{% endif %}> Use Demo Image<br>
  <input type=radio name=img_mode value="upload" {% if img_mode != 'demo' %}checked{% endif %}> Upload Image<br><br>
  <div id="upload-fields" {% if img_mode == 'demo' %}style="display:none;"{% endif %}>
    <label>Upload Input Image:</label><br>
    <input type=file name=file accept="image/*"><br><br>
    <input type="checkbox" id="use-template" name="use_template" {% if use_template %}checked{% endif %}>
    <label for="use-template">Upload Template Image (for alignment)</label><br>
    <div id="template-upload" style="display: {% if use_template %}block{% else %}none{% endif %};">
      <label>Upload Template Image:</label><br>
      <input type=file name=template_file accept="image/*"><br><br>
    </div>
  </div>
  <input type=hidden name=action value="upload">
  <button type="submit" id="upload-btn">Upload</button>
</form>
<script>
document.addEventListener('DOMContentLoaded', function() {
  function updateFields() {
    var mode = document.querySelector('input[name="img_mode"]:checked').value;
    document.getElementById('upload-fields').style.display = (mode === 'demo') ? 'none' : '';
  }
  function updateTemplate() {
    var cb = document.getElementById('use-template');
    document.getElementById('template-upload').style.display = cb.checked ? 'block' : 'none';
  }
  var radios = document.querySelectorAll('input[name="img_mode"]');
  radios.forEach(function(r) { r.addEventListener('change', updateFields); });
  var cb = document.getElementById('use-template');
  if (cb) cb.addEventListener('change', updateTemplate);
  updateFields();
  updateTemplate();

  var uploadForm = document.getElementById('upload-form');
  if (uploadForm) {
    uploadForm.addEventListener('submit', function(e) {
      var btn = document.getElementById('upload-btn');
      if (btn) {
        btn.disabled = true;
        btn.innerText = 'Uploading image...';
      }
    });
  }
  var detectForm = document.getElementById('detect-form');
  if (detectForm) {
    detectForm.addEventListener('submit', function(e) {
      var btn = document.getElementById('detect-btn');
      if (btn) {
        btn.disabled = true;
        btn.innerText = 'Running text detection...';
      }
    });
  }
});
</script>
{% if result_url %}
  <h2>Scanned Result{% if template_url %} & Template{% endif %}:</h2>
  <div style="display:flex; gap:30px; align-items:flex-start;">
    <div>
      <b>Scanned Result</b><br>
      <img src="{{ result_url }}" style="max-width: 350px; border:1px solid #aaa;">
      <br><a href="{{ result_url }}" download>Download Result</a>
    </div>
    {% if template_url %}
    <div>
      <b>Template Image</b><br>
      <img src="{{ template_url }}" style="max-width: 350px; border:1px solid #aaa;">
      <br><a href="{{ template_url }}" download>Download Template</a>
    </div>
    {% endif %}
  </div>
  <form method=post style="margin-top:20px;" id="detect-form">
    <p>Specify which fields should be detected. The model will attempt to find the given text field and will capture any values to the right within the given margin or 600px if no margin is specified.</p>
    <label>Enter OCR Fields (one per line, format: text[,margin(px)]):</label><br>
    <textarea name="ocr_fields" rows="6" cols="40">{{ ocr_fields_text }}</textarea><br>
    <input type=hidden name=action value="detect_text">
    <input type=hidden name=result_path value="{{ result_path }}">
    <input type=hidden name=img_mode value="{{ img_mode }}">
    <input type=hidden name=template_path value="{{ template_path }}">
    <button type="submit" id="detect-btn">Detect Text & Extract Fields</button>
  </form>
{% endif %}
{% if annotated_url %}
  <h2>Keyword Fields:</h2>
  <img src="{{ annotated_url }}" style="max-width: 500px;">
  <br><a href="{{ annotated_url }}" download>Download Annotated Image</a>
{% endif %}
{% if keyword_fields_table %}
  <h2>Extracted Fields Table:</h2>
  {{ keyword_fields_table|safe }}
  <form method="get" action="/export_csv" style="margin-top:10px;">
    <button type="submit">Export as CSV</button>
  </form>
{% endif %}
'''
    
import os

@app.route('/', methods=['GET', 'POST'])
def index():
    result_url = None
    result_path = None
    template_path = None
    template_url = None
    keyword_fields = None
    annotated_url = None
    keyword_fields_table = None
    ocr_fields_text = '\n'.join([f"{loc.text}" for loc in DEFAULT_OCR_LOCATIONS])
    OCR_LOCATIONS = DEFAULT_OCR_LOCATIONS
    img_mode = 'upload'
    use_template = False
    from flask import flash, session
    if request.method == 'POST':
        action = request.form.get('action', 'upload')
        img_mode = request.form.get('img_mode', 'upload')
        use_template = bool(request.form.get('use_template'))
        if action == 'upload':
            flash('Uploading image...')
            try:
                flash('Scanning image...')
                if img_mode == 'demo':
                    use_template=True
                    demo_img_path = 'test_img/IMG_7515_annotated_3.jpeg'
                    demo_template_path = 'test_img/template.jpeg'
                    img = Image.open(demo_img_path).convert('RGB')
                    img_np = np.array(img)
                    template_img = Image.open(demo_template_path).convert('RGB')
                else:
                    if 'file' not in request.files or request.files['file'].filename == '':
                        flash('Input image must be uploaded.')
                        return redirect(request.url)
                    file = request.files['file']
                    img = Image.open(file.stream).convert('RGB')
                    img_np = np.array(img)
                    if use_template:
                        if 'template_file' not in request.files or request.files['template_file'].filename == '':
                            flash('Template image must be selected if template alignment is enabled.')
                            return redirect(request.url)
                        template_file = request.files['template_file']
                        template_img = Image.open(template_file.stream).convert('RGB')
                if use_template:
                    cropped = scan(img_np, model)
                else:
                    cropped = img_np
                cropped_img = Image.fromarray(cropped.astype(np.uint8))
                buf = io.BytesIO()
                cropped_img.save(buf, format='PNG')
                buf.seek(0)
                os.makedirs('static', exist_ok=True)
                cropped_img.save('static/result.png')
                result_url = url_for('static', filename='result.png')
                result_path = 'static/result.png'
                if img_mode == 'demo' and use_template:
                    Image.open(demo_template_path).save('static/demo_template.png')
                    template_path = 'static/demo_template.png'
                    template_url = url_for('static', filename='demo_template.png')
                elif img_mode != 'demo' and use_template:
                    template_img.save('static/template.png')
                    template_path = 'static/template.png'
                    template_url = url_for('static', filename='template.png')
                else:
                    template_path = None
                    template_url = None
                flash('Image uploaded and scanned. Ready for text detection.')
                ocr_fields_text = '\n'.join([f"{loc.text}" for loc in DEFAULT_OCR_LOCATIONS])
            except Exception as e:
                flash('Error during upload/scanning.')
                flash(f'Processing failed: {e}')
        elif action == 'detect_text':
            flash('Finding text and extracting fields...')
            result_path = request.form.get('result_path', None)
            template_path = request.form.get('template_path', None)
            ocr_fields_text = request.form.get('ocr_fields', ocr_fields_text)
            img_mode = request.form.get('img_mode', 'upload')
            use_template = bool(request.form.get('use_template'))
            OCR_LOCATIONS = []
            for line in ocr_fields_text.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                print(parts)
                if len(parts) >= 1 and parts[0]:
                    text = parts[0]
                    id_val = text.lower().replace(' ', '_')
                    margin = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 600
                    OCR_LOCATIONS.append(OCRLoc(id_val, text, margin))
                print(OCR_LOCATIONS)
            try:
                if result_path and os.path.exists(result_path):
                    cropped_img = Image.open(result_path).convert('RGB')
                    cropped = np.array(cropped_img)
                    keyword_fields, annotated_img = get_keyword_fields(cropped, OCR_LOCATIONS)
                    annotated_pil = Image.fromarray(annotated_img.astype(np.uint8))
                    buf_ann = io.BytesIO()
                    annotated_pil.save(buf_ann, format='PNG')
                    buf_ann.seek(0)
                    temp_ann_path = 'static/annotated.png'
                    os.makedirs('static', exist_ok=True)
                    annotated_pil.save(temp_ann_path)
                    annotated_url = url_for('static', filename='annotated.png')
                    table_html = '<table border="1" cellpadding="4"><tr><th>Keyword</th><th>Extracted Text</th><th>Score</th></tr>'
                    for kw, items in keyword_fields.items():
                        for t, box, score in items:
                            table_html += f'<tr><td>{kw}</td><td>{t}</td><td>{score:.2f}</td></tr>'
                    table_html += '</table>'
                    keyword_fields_table = table_html
                    result_url = url_for('static', filename='result.png')
                    flash('Text detection and extraction complete.')
                else:
                    flash('No scanned image found for text detection.')
            except Exception as e:
                flash('Error during text detection.')
                print(f'Processing failed: {e}')
                flash(f'Processing failed: {e}')
    if keyword_fields is not None:
        session['keyword_fields_csv'] = {k: [t for t, box, score in v] for k, v in keyword_fields.items()}
    return render_template_string(HTML, result_url=result_url, result_path=result_path, template_path=template_path, template_url=template_url, annotated_url=annotated_url, keyword_fields_table=keyword_fields_table, ocr_fields_text=ocr_fields_text, img_mode=img_mode, use_template=use_template)

@app.route('/export_csv')
def export_csv():
    keyword_fields = session.get('keyword_fields_csv', {})
    if not keyword_fields:
        return Response('No data to export.', mimetype='text/plain')
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Field', 'Extracted Texts'])
    for field, texts in keyword_fields.items():
        writer.writerow([field, str(texts)])
    output.seek(0)
    return Response(output.read(), mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename=extracted_fields.csv'
    })

if __name__ == '__main__':
    app.run(debug=True)