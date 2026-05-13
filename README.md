# Face Recognition Using OpenCV

A face recognition project that identifies Indian cricketers (Virat Kohli, Rohit Sharma, and MS Dhoni) using OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer.

## Project Structure

```
FaceRecognitionUsingOPENCV/
├── Face_recognizer.py       # Recognizes faces using the trained model
├── face_training.py         # Trains the face recognition model
├── face_trainedmodel.yml    # Saved trained model
├── features.npy             # Extracted face features
├── labels.npy               # Corresponding labels
└── FaceRecognitionImages/   # Training images organized by person
    ├── ViratKohli/
    ├── RohitSharma/
    └── Dhoni/
```

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python` and `opencv-contrib-python`)
- NumPy

## Installation

```bash
pip install opencv-python opencv-contrib-python numpy
```

## Usage

### 1. Training the Model

The training script reads images from `FaceRecognitionImages/` (organized in subfolders named after each person), detects faces using Haar Cascade, and trains an LBPH recognizer.

**Note:** Update the hardcoded paths in `face_training.py` with your own paths:

```python
# Path to the FaceRecognitionImages folder on your machine
DIR = r'YOUR_PATH\FaceRecognitionUsingOPENCV\FaceRecognitionImages'

# Path to Haar Cascade XML (or use OpenCV's built-in one)
haar_cascade = cv.CascadeClassifier('YOUR_PATH\\haarcascade_frontalface_default.xml')
# Alternative — use OpenCV's built-in cascade:
# haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

Then run:
```bash
python face_training.py
```

This will generate `face_trainedmodel.yml`, `features.npy`, and `labels.npy`.

### 2. Recognizing Faces

The recognizer script loads the trained model and predicts faces in a test image.

Update the test image path in `Face_recognizer.py` with your own path:

```python
img = cv.imread(r"YOUR_PATH\TestImage.jpg")
```

Then run:
```bash
python Face_recognizer.py
```

The script will display the image with bounding boxes and labels identifying detected faces.

## How It Works

1. **Face Detection**: Uses Haar Cascade classifier to detect faces in images.
2. **Feature Extraction**: Extracts regions of interest (ROI) from detected faces.
3. **Training**: LBPH face recognizer is trained on the extracted face features.
4. **Recognition**: New faces are compared against the trained model and the best match is returned with a confidence score.

## Adding New People

1. Create a new folder inside `FaceRecognitionImages/` named after the person.
2. Add their face images (preferably well-lit, front-facing).
3. Add the person's name to the `people` list in both Python files.
4. Retrain the model using `face_training.py`.
