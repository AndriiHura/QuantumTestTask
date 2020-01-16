# QuantumTestTask

Solution for Data Science Bowl 2018 (https://www.kaggle.com/c/data-science-bowl-2018/overview)
The main objective was to build a model based on Unet Architeture for image semantic segmentation.

Model was build using Keras, and Tensorflow backend.
___

# Requirements

All requirements that you need are in Requirments.txt

Also you python 3.7 interpreter, and you need jupyter notebook.

# Overview

1.ExploratoryDataAnalysis.ipynb

  I did a basic EDA for our test and train dataset. Found out that for each train image there are a whole bunsh of masks, and they has to be composed into one, that contains all the features.

___

2.Predict.py

  Script that run model to predict on test data. It can use already trained weights, or create new one by running Train.py.

___

3. Train.py 
  
  Script runs model training on train data set. I recommend to use already trained weights, because unless you have powerful hardware, it will take you a lot of time to train model.

___

4. model.py

  Implementation of U-Net model itself, and dice_coef function to estimate accuracy of training.
 
___
 
5. model_weights.h5
  
  This file contains already trained weights for a model. I did training using kaggle kernels as they allow use GPU, and it boosts training in about 100 times, comparing to by laptop.
  
___

6. read_data.py

  Script, that contains functionality to read train and test data.
  
___

7. train_data, test_data

  Folders that contain data needed for training and predicting
  
___

8. predicted_test_data_gray, predicted_test_data_viridis

  Folders that contain test images, and predicted masks for that images. The only difference between these 2 folders is colormap of predicted masks.
  
___
  
9. Preliminary notebooks

  Folder that contains notebooks where I tested if scripts are doing well.
  
___

# Link to the kaggle kernel
  
  [a link](https://www.kaggle.com/andriihura/imagesegmentation)

  Here is a link to a kaggle kernel, where I tested different things, from adjusting hyperparameters of model, to changing layers' structure.
  Unfortunatel, as I was using this kernel as a draft, my code there is really super messy, and looks like a trash. So better don't check it :)
  

