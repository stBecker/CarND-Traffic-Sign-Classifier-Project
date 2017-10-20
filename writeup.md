# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[hist_train]: ./writeup/hist_train.png "Histogram of training data"
[hist_test]: ./writeup/hist_test.png "Histogram of test data"
[hist_valid]: ./writeup/hist_valid.png "Histogram of valid data"

[10]: ./writeup/10.png "Example traffic sign images"
[26]: ./writeup/26.png "Example traffic sign images"
[30]: ./writeup/30.png "Example traffic sign images"

[32gray]: ./writeup/32gray.png "Example traffic sign image in grayscale"

[hist_balanced]: ./writeup/hist_balanced.png "Histogram of the balanced training data"

[plot_accuracy]: ./writeup/plot_accuracy.png "Plot of the training/validation set accuracy"

[2]: ./writeup/2.png "Traffic Sign 2"
[13]: ./writeup/13.png "Traffic Sign 13"
[14]: ./writeup/14.png "Traffic Sign 14"
[15]: ./writeup/15.png "Traffic Sign 15"
[28]: ./writeup/28.png "Traffic Sign 28"

[bar_chart_prediction_softmax]: ./writeup/bar_chart_prediction_softmax.png "Top 5 softmax probabilities"
[recall_prec_bar]: ./writeup/recall_prec_bar.png "Recall and precision"

[conv1_i1]: ./writeup/conv1_i1.png "Conv1 input"
[conv1_i2]: ./writeup/conv1_i2.png "Conv1 input"
[conv1_o1]: ./writeup/conv1_o1.png "Conv1 output"
[conv1_o2]: ./writeup/conv1_o2.png "Conv1 output"
[conv2_o1]: ./writeup/conv2_o1.png "Conv2 output"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stBecker/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![alt text][hist_train]

We can see that the traffic signs have an unequal number of examples: some traffic signs are as many as ~2000 times in the dataset (e.g. 2 and 13), some traffic signs only have ~200 examples (e.g. 0 and 32) - an order of magnitude difference!
Below we have the distributions for the validation and test data, which is approximately the same as for the training data.

![alt text][hist_valid]

![alt text][hist_test]

Rendering some randomly selected example images from the training set we can visualize the data.

![alt text][10]
![alt text][26]
![alt text][30]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1)  The image data is normalized to zero mean and sigma=1, to make the training more efficient and make it easier to include new test samples. Normalizing the image does not change the appearance of the image.

2) The data was grayscaled in an attempt to improve the validation accurancy; ultimately we decided against grayscaling, because this causes valuable information to be lost,
e.g. lots of red pixels make the image llikely to be a stop sign, whereas lots of blue pixels would decrease the likelihood of a stop sign.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][32gray]


We decided to generate additional data because - as seen above - the training data is very unevenly distributed: there are many more examples of some traffic signs than there are of some others. Thus the CNN is probably overfitting to these traffic signs.

To make the distribution more even, we sampled exactly 2000 images for each traffic sign class from the training data.

    for class_id in range(n_classes):
        idx = dfy[dfy[0] == class_id].sample(2000, replace=True).index
        x = X_train[idx]
        y = y_train[idx]
        assert np.unique(y).size == 1
        train_data_balanced.append((y, x))

Below is a histogram of the resulting new distribution.

![alt text][hist_balanced]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid  padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid,  outputs 14x14x6				|
| Convolution 5x5	    | 1x1 stride, valid,  outputs 10x10x16   									|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid,  outputs 5x5x16				|
| Fully connected		| input 400, output 120 									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input 120, output 84 									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input 84, output 43 									|
| Softmax				|       									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, we used:
* the Adam-Optimizer,
* a learning rate of 0.001,
* 10 epochs with a batch size of 128,
* mu = 0 and sigma = 0.1 for the truncated normal distribution of the weights,
* a keep probability of 0.5 for the dropout,
* the balanced training data described above (86,000 images, 2,000 from each traffic sign class).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.961
* test set accuracy of 0.938

We started with the vanilla LeCun CNN from the lecture, only adapting the output vector to the different number of labels.
Changing the number of epochs and the batch size did not improve the validation accuracy to a satisfactory level, so we tried to feed grayscaled images to the model.
Using grayscaled data  did not yield better results, presumably because too much valuable information was lost by discarding the color channels of the images.
Finally we added dropout-layers to the fully connected layers of the models to reduce overfitting, which finally resulted in a good validation set accuracy.


![alt text][plot_accuracy]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][2] ![alt text][13] ![alt text][14]
![alt text][15] ![alt text][28]

The images are all rather different from the training examples, as they depict only the traffic sign, without any noise, distortions or background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 50 km/h	      		| 30 km/h					 				|
| Yield					| Yield											|
| Stop Sign      		| Stop Sign   									|
| No vehicles      		| No vehicles   									|
| Children crossing      		| Children crossing   									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.938

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Plotting the top 5 softmax probabilities for the traffic sign predictions, we can see that that the CNN is very certain about its
predictions, with probabilities of close to 100% for all but 50 km/h sign (class 2) - which is also the only false prediction. The second highest probability of  ~ 1% for 50 km/h sign is given to the correct class.


![alt text][bar_chart_prediction_softmax]


real class: 13
1. predicted class 13 probability 1.0000
2. predicted class 0 probability 0.0000
3. predicted class 1 probability 0.0000
4. predicted class 2 probability 0.0000
5. predicted class 3 probability 0.0000

real class: 14
1. predicted class 14 probability 0.9988
2. predicted class 25 probability 0.0012
3. predicted class 17 probability 0.0000
4. predicted class 26 probability 0.0000
5. predicted class 15 probability 0.0000

real class: 15
1. predicted class 15 probability 0.9989
2. predicted class 9 probability 0.0004
3. predicted class 26 probability 0.0004
4. predicted class 22 probability 0.0001
5. predicted class 29 probability 0.0001

real class: 2
1. predicted class 1 probability 0.9886
2. predicted class 2 probability 0.0113
3. predicted class 4 probability 0.0001
4. predicted class 7 probability 0.0000
5. predicted class 0 probability 0.0000

real class: 28
1. predicted class 28 probability 1.0000
2. predicted class 23 probability 0.0000
3. predicted class 20 probability 0.0000
4. predicted class 41 probability 0.0000
5. predicted class 36 probability 0.0000


Recall and precision:

![alt text][recall_prec_bar]

The model seems to have the biggest problems in correctly classifying the Double curve sign and the Pedestrians sign.
We can see that in the test set the recall and precision for both Speed limit (30km/h) and Speed limit (50km/h) are fairly high.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first convolutional layer clearly activates on the rough shape of the traffic sign:

##### Example 1: Detected round shape
![alt text][conv1_i1]
![alt text][conv1_o1]

##### Example 2: Detected triangle shape
![alt text][conv1_i2]
![alt text][conv1_o2]

The second convolutional layer is not as easily interpretable as the first one, no clear pattern becomes obvious when observing the output:

##### Example 3: Random looking pattern
![alt text][conv1_i2]
![alt text][conv2_o1]