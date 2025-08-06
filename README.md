# Document Scanner

This document scanner aims to provide a no-frills, setup-free automatic document scanner for data entry. It detects text using PaddleOCR and automatically retrieves adjacent text to provided field names, allowing data extraction out-of-the-box without prior annotation.

## How to use

1. Add an image of a form (or use the default image for demo purposes). The image will be automatically aligned, so odd angles/skewing to a reasonable extent shouldn't be an issue!
2. Specify which fields should be scanned for. This should be the label that appears directly to the left of the data you wish to capture. See the default value (matched to the default image) for an example.
3. Submit and wait for the scan to complete! If using the Gradle demo, the extracted fields will appear on the left side of the screen. If you're running the program locally, you can see results at the bottom of the page and export them as a CSV.

## Running Locally (Flask App)

1. **Clone the repository:**
   ```
   git clone https://github.com/giragon6/porch-form-scanner.git
   cd porch-form-scanner
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```
   python main.py
   ```

4. **Visit the app:**
   - Go to [http://localhost:5000](http://localhost:5000)

You can now upload your own images (or use the demo image), specify fields, and extract data from your forms!

---
## Features

### Current
- Automatic image scanning and alignment
- Form-agnostic field extraction (no setup)

### Planned
- Speed optimizations via model downscaling and image preprocessing
- Table cell recognition
- Support for vertically aligned labels

## Attribution

The following tutorials were used to get a starting point but were heavily modified:

I used the dataset and model from LearnOpenCV's [Document Segmentation Using Deep Learning in PyTorch post](https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/) for the document-scanning portion of this project.

I adapted code from [pyimagesearch](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/) for aligning documents to a template image.