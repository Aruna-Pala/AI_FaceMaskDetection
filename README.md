# Project : AI Face Mask Detector
 COMP 6721- Applied Artificial Intelligence

## Prerequisites
- Python 3.8

## Setting Up a Project
1. Run the TrainFaceMaskCNN.py file with the correct data set zip file path
2. After executing the TrainFaceMaskCNN.py, a file will be generated with trained model information that can be used for predicting the images in prediction folder along with Loss-graph and Confusion matrix and a classification report csv file in the results folder.
3.After successful execution of step 1 run FaceMaskPrediction.py to predict all the images that are present in given prediction folder path.
4. For all the images in prediction folder it will predict the image into either one of the categories from Cloth_Mask, No_Mask, Surgical_Mask,N95_Mask, Mask_Worn_incorrectly. 
5. The model will run on the bias evaluation dataset as well. 
6. All the results will be saved to Result/Prediction folder

**Please find the Dataset link below(as the file is large we added the link here)**
## Data Set Link :- 
https://liveconcordia-my.sharepoint.com/:u:/g/personal/d_dantu_live_concordia_ca/EXhlgd6ys01IvczioF-CtoABe0B2Vqn_QArn5yJPw29xcQ?e=4rX7JF

## Biased Data Set Link :- 
https://liveconcordia-my.sharepoint.com/:u:/g/personal/d_dantu_live_concordia_ca/EcXx9LJn7e5JpN8fBZ7usPMBDICjnu31_OL3cSnU6VXexg?e=b1epyC

