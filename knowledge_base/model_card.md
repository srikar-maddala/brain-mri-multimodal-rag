---
title: Model Card - Brain MRI Classifier and Retriever (this project)
source: This project's own evaluation results
url: local
topic: model
---

# Model Card - Brain MRI Classifier and Retriever

## Intended use and disclaimer
This system is an educational and research demo. It is NOT a medical device and must NOT be used for diagnosis or treatment decisions. Any MRI finding must be interpreted by a qualified radiologist or physician, and a definitive tumor diagnosis usually requires a biopsy. Under the EU Medical Device Regulation (MDR), software intended for diagnosis would require clinical validation and certification, which this project does not have.

## Model and training data
The classifier is a ResNet18 pretrained on ImageNet and fine-tuned on the public Kaggle "Brain Tumor Classification (MRI)" dataset with four classes: glioma_tumor, meningioma_tumor, no_tumor and pituitary_tumor. Training used 2,870 images (826 glioma, 822 meningioma, 395 no tumor, 827 pituitary) with augmentation (flips, rotation, affine shifts, brightness and contrast jitter) and class-weighted cross-entropy loss. Image retrieval uses the 512-dimensional ResNet18 embeddings plus a supervised contrastive (SupCon) projection head, with cosine similarity against the training images.

## Performance
Validation accuracy on a split of the training folder was about 97 percent, but test accuracy is much lower, showing a strong distribution shift between the dataset's Training and Testing folders. After removing duplicate images, test accuracy is 69.3 percent. Glioma recall is only about 24 percent: most glioma test images are misclassified as meningioma or no tumor. Retrieval precision at 5 (share of the top 5 similar cases with the same label) is 0.679 with ResNet embeddings and 0.701 with the SupCon head.

## Data leakage found
An MD5 hash check found exact duplicate images between the Training and Testing folders: 88 test images (all labeled no_tumor) were copies of training images. Including them inflated test accuracy from 69.3 to about 76 percent. All reported results remove these duplicates. After removal only 17 no_tumor test images remain, so per-class results for no_tumor are unreliable.

## Robustness and known failure modes
Robustness was tested with six image corruptions at three severities. The model is robust to brightness, contrast and rotation changes (within about 5 points of clean accuracy), which were part of training augmentation. It is sensitive to JPEG compression (about 6 points lower) and blur (about 25 to 30 points lower). It fails completely under Gaussian noise: at noise level 0.1 accuracy falls to 5.6 percent because the model predicts no_tumor for every image. A model that answers "no tumor" for low-quality images is a dangerous failure mode, so predictions on noisy or blurred scans must not be trusted.

## How to interpret a prediction
Treat the predicted class as a hint, not a diagnosis. Check the confidence score, whether the similar retrieved cases agree with the prediction, and the Grad-CAM heatmap. Low confidence, disagreement between the classifier and the retrieved cases, or a glioma-like image predicted as meningioma or no tumor are warning signs. A "no tumor" prediction does not rule out a tumor or any other brain condition.
