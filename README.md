# Skin-image-Classification
Introduction
This README file provides an overview of the code for skin image classification using Mask R-CNN. This code allows you to train a Mask R-CNN model for lesion boundary detection and make predictions on skin images.

# Abstract 
Skin cancer is a prevalent condition among white-skinned people in their middle age or older. Although malignant melanoma is one of the most dangerous types of cancer, it can be treated with immediate care.Here,a method is presented for segmenting the lesion in a precise, automated manner in terms of the ease of classification and achieve high accuracy during the initial phases of skin cancer diagnosis. The proposed method utilizes the Mask RCNN approach to not only detect but also segment the region of interest in terms of pixels, making this approach very precise in detecting the cancer region with an accurate boundary. The Proposed method will be trained on ISIC Skin Lesion dataset that contains input images and masks for the same. The trained Mask RCNN has the capability to segment the cancer region and once it is segmented the region will be further cropped for cancer classification.

# Statement of Purpose 
The purpose of this project isto perform theConvolutional Neural Network(Convolutional Neural Network (CNN)) to diagnose skin lesions.Here Mask Regions-based Convolutional Neural Networks (R-CNN) which is in terms of image segmentation, used to segment the affected region and its further processed for cancer classification. To achieve all the process, we have to:
✼ We develop the algorithm for prediction of skin lesion segmentation from the input images and with its respective binary masks.
✼ The images and binary masks should be processed and detection is done pixel by pixel manner.
✼ The output is produced by segmenting the skin by the mask overlaid on the input image with good accuracy.

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
![Leison Image](https://github.com/pradhip11/Skin-image-Classification/assets/148735328/2ec687f3-3c28-41d4-9b9d-4ed7b67477ef)

![Mask Image](https://github.com/pradhip11/Skin-image-Classification/assets/148735328/30f67402-3878-4790-9c70-fa2bd646b21d)



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
   
  # Conclusion
   In this paper, we have developed an automated method for segmenting lesions and accurately classifying them with high accuracy. Here, Mask R-CNN approach is  
   usednot only to detect but also helps in segmenting the accurate lesion boundary. By providing a new way for evaluating abrupt cutoff and improving the 
   performance of feature extraction algorithms, a novel and effective strategy for eliminating subjectivity in visual interpretation of dermoscopy images and 
   reducing the incidence of false negative/false positive diagnoses is provided. The proposed model has a 90 percent accuracy rate.This will be a huge benefit in 
   diagnosing skin cancer in the early stage of clinical practice.

   ![final output](https://github.com/pradhip11/Skin-image-Classification/assets/148735328/6e75af1a-e3fb-41c6-9678-791fab3278f9)

  
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


