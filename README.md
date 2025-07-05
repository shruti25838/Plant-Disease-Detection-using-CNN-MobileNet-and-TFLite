# Plant Disease Detection using CNN, MobileNet, and TFLite

This project involves detecting plant diseases using a combination of CNNs, MobileNet transfer learning, TensorFlow Lite, and Raspberry Pi camera integration.

## Features

- CNN-based custom model for initial classification
- MobileNet + Dense layers for improved transfer learning
- TFLite deployment for lightweight edge inference
- PiCamera integration for real-time image capture

## Model Architectures

### 1. CNN Model (`cnn_model.py`)
- Convolutional layers → MaxPooling → Fully Connected layers
- Optimized with Adam and trained for 5 epochs
- Uses `binary_crossentropy` for two-class disease classification

### 2. MobileNet Model (`disease_model.py`)
- Pre-trained MobileNet used as a feature extractor
- Followed by custom Dense layers with Dropout
- Optimized using categorical cross-entropy

### 3. TFLite Inference (`tflite_inference.py`)
- Loads a `.tflite` model
- Preprocesses test image using PIL
- Predicts label and prints classification result

## Raspberry Pi Camera

- Script `pi_camera.py` captures an image using the PiCamera module.
- Image is saved and used for real-time disease detection on-device.

## Files

| File                | Description                               |
|---------------------|-------------------------------------------|
| `cnn_model.py`      | CNN training code                         |
| `disease_model.py`  | MobileNet transfer learning code          |
| `tflite_inference.py`| TensorFlow Lite classification script     |
| `pi_camera.py`      | PiCamera image capture script             |

## Requirements
pip install tensorflow keras numpy pillow tflite-runtime
