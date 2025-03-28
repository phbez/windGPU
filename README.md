# windGPU
Wind Turbine Blade Fault Detection Model

## Overview
A CNN-based model for detecting faults in small wind turbine blades using image classification.

### Model Architecture
- Input: 48x48 RGB images
- Layers:
  - BatchNormalization
  - Conv2D (512 filters) + ReLU
  - MaxPooling2D
  - Dropout (0.2)
  - Conv2D (256 filters) + ReLU
  - MaxPooling2D
  - Dropout (0.2)
  - Dense (32 units)
  - Dense (2 units, softmax)

### Performance
- Training accuracy: 0.9676
- Validation accuracy: 0.9745

### Dataset
- Classes: Faulty, Healthy
- Image size: 48x48 pixels
- Color space: RGB

## Usage
```python
from keras.models import load_model

# Load model
model = load_model('modeloCAI.keras')

# Preprocess image (48x48 RGB)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict
prediction = model.predict(preprocessed_image)
```

## Files Included
- `modeloCAI.keras` - Trained model weights and parameters
- `CAImodel.json` - Model architecture configuration
- `datasetCAI.csv` - Training history and metrics data

## Requirements
- TensorFlow 2.x
- OpenCV
- NumPy
- Keras
