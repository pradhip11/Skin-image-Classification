# Skin-image-Classification
Introduction
This README file provides an overview of the code for skin image classification using Mask R-CNN. This code allows you to train a Mask R-CNN model for lesion boundary detection and make predictions on skin images.

# Setup
Before using the code, please make sure to set up your environment. The code is designed to run on Google Colab, and it requires several Python packages. To install these packages and mount your Google Drive for data access, use the following code:

from google.colab import drive
drive.mount('/content/drive')

! pip uninstall -y tensorflow keras

! pip install mrcnn==0.1 opencv-python==3.4.2.17 tensorflow-gpu==1.13.1 keras==2.1.5

!pip install h5py==2.10.0

# Code Overview
The code is divided into two modes: "Train" and "Predict." Let's break down each section:

## Train Mode
If you intend to train a Mask R-CNN model for lesion boundary detection, set mode = "Train" in the code. The training process involves the following steps:

Data Preparation: You need to specify the dataset path, images path, and annotations file path. The code randomly splits the dataset into training and validation sets.

Configuration: Define the configuration for the Mask R-CNN model. This includes specifying the number of GPUs to use, images per GPU, steps per training epoch, and other hyperparameters.

Dataset Loading: The code loads the training and validation datasets, preparing them for training. It also applies data augmentation techniques.

## Model Initialization: The Mask R-CNN model is initialized, and pre-trained COCO weights are loaded for fine-tuning.

Training: The model is trained in two stages. First, only the layer heads are trained, and then the entire network is fine-tuned. The number of epochs and learning rates are specified.

## Predict Mode
If you want to use a trained Mask R-CNN model to make predictions on skin images, set mode = "predict" in the code. The prediction process involves the following steps:

Here are the key steps for using the code in "Predict" mode in bullet points:

- **Configuration**:
  - Initialize the inference configuration for the Mask R-CNN model.

- **Model Loading**:
  - Load the pre-trained Mask R-CNN model for inference.

- **Image Loading**:
  - Load an input image.
  - Convert it from BGR to RGB channel ordering.
  - Resize the image to the required dimensions.

- **Inference**:
  - Perform a forward pass of the model to detect lesions in the image.

- **Visualization**:
  - Visualize the detected lesions by:
    - Drawing bounding boxes around the lesions.
    - Displaying class labels.
  
- **Usage**:
  - Set the `mode` variable to "predict" in the code.
  - Make sure you have the necessary image and model files available.
  - Run the code in a compatible environment, such as Google Colab.


## To use the code, follow these steps:
Set the mode variable to either "Train" or "predict" depending on your task.
Modify the dataset paths, training parameters, and any other configurations as needed.
Run the code in Google Colab.

## Note
Ensure that you have the necessary dataset files and pre-trained COCO weights in your Google Drive for training. Adjust the paths accordingly.By following this README and the provided code, you can train a Mask R-CNN model for skin image classification or make predictions on skin images with existing models.


