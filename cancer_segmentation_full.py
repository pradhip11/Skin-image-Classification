from google.colab import drive
drive.mount('/content/drive')

! pip uninstall -y tensorflow keras

! pip install mrcnn==0.1 opencv-python==3.4.2.17 tensorflow-gpu==1.13.1 keras==2.1.5

!pip install h5py==2.10.0

# import the necessary packages
from imgaug import augmenters as iaa
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
from google.colab.patches import cv2_imshow

#mode = "Train"
mode = "predict"

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("/content/drive/MyDrive/isic2018")
IMAGES_PATH = os.path.sep.join([DATASET_PATH,
	"ISIC2018_Task1-2_Training_Input"])
MASKS_PATH = os.path.sep.join([DATASET_PATH,
	"ISIC2018_Task1_Training_GroundTruth"])

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.8

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "lesion"}

# initialize the path to the Mask R-CNN pre-trained on COCO

COCO_PATH = "/content/drive/MyDrive/cancer/lesions_logs/lesion20220324T1611/mask_rcnn_lesion_0014.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "/content/drive/MyDrive/cancer/lesions_logs"

class LesionBoundaryConfig(Config):
	# give the configuration a recognizable name
	NAME = "lesion"

	# set the number of GPUs to use training along with the number of
	# images per GPU (which may have to be tuned depending on how
	# much memory your GPU has)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the number of steps per training epoch and validation cycle
	STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
	VALIDATION_STEPS = len(valIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

	# number of classes (+1 for the background)
	NUM_CLASSES = len(CLASS_NAMES) + 1

class LesionBoundaryInferenceConfig(LesionBoundaryConfig):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9

class LesionBoundaryDataset(utils.Dataset):
	def __init__(self, imagePaths, classNames, width=1024):
		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.classNames = classNames
		self.width = width

	def load_lesions(self, idxs):
		# loop over all class names and add each to the 'lesion'
		# dataset
		for (classID, label) in self.classNames.items():
			self.add_class("lesion", classID, label)

		# loop over the image path indexes
		for i in idxs:
			# extract the image filename to serve as the unique
			# image ID
			imagePath = self.imagePaths[i]
			filename = imagePath.split(os.path.sep)[-1]

			# add the image to the dataset
			self.add_image("lesion", image_id=filename,
				path=imagePath)

	def load_image(self, imageID):
		# grab the image path, load it, and convert it from BGR to
		# RGB color channel ordering
		p = self.image_info[imageID]["path"]
		image = cv2.imread(p)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# resize the image, preserving the aspect ratio
		image = imutils.resize(image, width=self.width)

		# return the image
		return image

	def load_mask(self, imageID):
		# grab the image info and derive the full annotation path
		# file path
		info = self.image_info[imageID]
		filename = info["id"].split(".")[0]
		annotPath = os.path.sep.join([MASKS_PATH,
			"{}_segmentation.png".format(filename)])

		# load the annotation mask and resize it, *making sure* to
		# use nearest neighbor interpolation
		annotMask = cv2.imread(annotPath)
		annotMask = cv2.split(annotMask)[0]
		annotMask = imutils.resize(annotMask, width=self.width,
			inter=cv2.INTER_NEAREST)
		annotMask[annotMask > 0] = 1

		# determine the number of unique class labels in the mask
		classIDs = np.unique(annotMask)

		# the class ID with value '0' is actually the background
		# which we should ignore and remove from the unique set of
		# class identifiers
		classIDs = np.delete(classIDs, [0])

		# allocate memory for our [height, width, num_instances]
		# array where each "instance" effectively has its own
		# "channel" -- since there is only one lesion per image we
		# know the number of instances is equal to 1
		masks = np.zeros((annotMask.shape[0], annotMask.shape[1], 1),
			dtype="uint8")

		# loop over the class IDs
		for (i, classID) in enumerate(classIDs):
			# construct a mask for *only* the current label
			classMask = np.zeros(annotMask.shape, dtype="uint8")
			classMask[annotMask == classID] = 1

			# store the class mask in the masks array
			masks[:, :, i] = classMask

		# return the mask array and class IDs
		return (masks.astype("bool"), classIDs.astype("int32"))
  
if __name__ == "__main__":
	# check to see if we are training the Mask R-CNN
	if mode == "Train":
		# load the training dataset
		trainDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
		trainDataset.load_lesions(trainIdxs)
		trainDataset.prepare()

		# load the validation dataset
		valDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
		valDataset.load_lesions(valIdxs)
		valDataset.prepare()

		# initialize the training configuration
		config = LesionBoundaryConfig()
		config.display()

		# initialize the image augmentation process
		aug = iaa.SomeOf((0, 2), [
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.Affine(rotate=(-10, 10))
		])

		# initialize the model and load the COCO weights so we can
		# perform fine-tuning
		model = modellib.MaskRCNN(mode="training", config=config,
			model_dir=LOGS_AND_MODEL_DIR)
		model.load_weights(COCO_PATH, by_name=True,
			exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
				"mrcnn_bbox", "mrcnn_mask"])

		# train *just* the layer heads
		model.train(trainDataset, valDataset, epochs=20,
			layers="heads", learning_rate=config.LEARNING_RATE,
			augmentation=aug)

		# unfreeze the body of the network and train *all* layers
		model.train(trainDataset, valDataset, epochs=40,
			layers="all", learning_rate=config.LEARNING_RATE / 10,
			augmentation=aug)

	# check to see if we are predicting using a trained Mask R-CNN
	elif mode == "predict":
		# initialize the inference configuration
		config = LesionBoundaryInferenceConfig()

		# initialize the Mask R-CNN model for inference
		model = modellib.MaskRCNN(mode="inference", config=config,
			model_dir=LOGS_AND_MODEL_DIR)

		# load our trained Mask R-CNN
		weights = "/content/drive/MyDrive/cancer/lesions_logs/lesion20220324T1611/mask_rcnn_lesion_0014.h5"
		model.load_weights(weights, by_name=True)

		# load the input image, convert it from BGR to RGB channel
		# ordering, and resize the image
		image = cv2.imread("/content/drive/MyDrive/isic2018/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = imutils.resize(image, width=1024)
		image_orig = image.copy()

		# perform a forward pass of the network to obtain the results
		r = model.detect([image], verbose=1)[0]
		# loop over of the detected object's bounding boxes and
		# masks, drawing each as we go along
		for i in range(0, r["rois"].shape[0]):
			mask = r["masks"][:, :, i]
			image = visualize.apply_mask(image, mask,
				(1.0, 0.0, 0.0), alpha=0.5)
			image = visualize.draw_box(image, r["rois"][i],
				(1.0, 0.0, 0.0))
			#cv2_imshow(image)
		# convert the image back to BGR so we can use OpenCV's
		# drawing functions
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# loop over the predicted scores and class labels
		for i in range(0, len(r["scores"])):
			# extract the bounding box information, class ID, label,
			# and predicted probability from the results
			(startY, startX, endY, end) = r["rois"][i]
			classID = r["class_ids"][i]
			label = CLASS_NAMES[classID]
			score = r["scores"][i]

			# draw the class label and score on the image
			text = "{}: {:.4f}".format(label, score)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		# resize the image so it more easily fits on our screen
		image = imutils.resize(image, width=512)
		image_orig = imutils.resize(image_orig, width=512)

		# show the output image
		cv2_imshow(image_orig)
		cv2_imshow(image)

		#cv2.waitKey(0)