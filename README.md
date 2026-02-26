# COSC465-Robotics CNN Waste Classifier
 We used the dataset from kaggle and downsized the original 10 classes, to 3 classes. Using a Convoluional Neural Network Model, we were able to classify an object as trash, recyclable, or compost. 

## Categories
### ♻️ Recycle
- battery 
- cardboard
- glass
- metal
- paper
- plastic

### 🌱 Compost
- biological (food waste, organic matter)

### 🗑️ Trash
- clothes 
- shoes 
- trash (general non-recyclable waste)


To get results on a live camera we utilized openCV, along with a pretrained detection model to draw bounding boxes around objects. Combining it with our CNN, we were then able to provide a label for each object.

Dataset can be found on kaggle: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data

## Model Architecture

### CNN Architecture (32×32 RGB Input)
```
Input: 3×32×32 (3 Color Channels, Image Tranformed into 32x32)
→ Conv(3→32) → ReLU → MaxPool
→ Conv(32→64) → ReLU → MaxPool
→ Flatten (4096)
→ FC(4096→128) → ReLU
→ FC(128→3)
```

### DNN Architecture
```
Pretrained Detection Model (OpenCV DNN)

# Load pretrained object detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Resize all input images to 320x320 (model expected input size)
net.setInputSize(320, 320)

# Normalize pixel values (scales 0–255 → approximately -1 to 1)
net.setInputScale(1.0 / 127.5)

# Subtract mean from each channel for centering
net.setInputMean((127.5, 127.5, 127.5))

# Convert OpenCV's default BGR format to RGB
net.setInputSwapRB(True)
```

## Files
```
waste-classifier-project/
├── savedModels/ # Save your models here    
├── soruce/ 
    ├── main_image.py # Tests model on one singular Image
    ├── webcam.py    # Tests the model in real time
├── cnntrainingcpu.ipynb      # Code to train a CNN on CPU
├── cnntraininggpu.ipynb      # Code to train a CNN on GPU
├── objectdetectioncv.yml     # Dependencies for the detection model
├── originalImages.zip        # Dataset (Recycle, Trash, Compost)
└── config_files.zip          # Configuration files
```

## Steps to Test Our Model in Real Time
### Install Dependencies
- **pip install requirements.txt**
- **installDatasets_configfiles.py**
   
### Unzip the Datasets/Configurations For Detection Model

- **unzip -r originalImages.zip**

- **unzip -r config_files.zip**

### Training/Evaluating on GPU/CPU
- **Run cnntraininggpu.ipynb (If cuda is available)**
- **If cuda is not available, Run cnntrainingcpu.ipynb** 

### Live Detection 
- **In source/webcam.py, modify model_path**
- **Run the file**

### Singular Photo Detection


## Results
| Epochs | Training Accuracy | Testing Accuracy | Device|  Average Loss Per Epoch|
| ----------- | ----------- | ----------- |----------- | -----------|
| 30 | 98% | 75% | GPU | 15%| 
| 50 | ... | ... | GPU | ...|
| 75 | ... | ... | GPU | ... |
| 100 | ... | ... | GPU | ...|
