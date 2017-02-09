##**Traffic Sign Recognition** 

##Writeup

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

[image1]: ./examples/visualization.png "Visualization"
[image4]: ./new_images/1.jpeg "Traffic Sign 1"
[image5]: ./new_images/3.jpeg "Traffic Sign 2"
[image6]: ./new_images/14.jpeg "Traffic Sign 3"
[image7]: ./new_images/16.jpeg "Traffic Sign 4"
[image8]: ./new_images/35.jpeg "Traffic Sign 5"
[image9]: ./new_images/38.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the forth and fifth code cell of the IPython notebook.  

One image of every class is shown in the forth code cell. The title of every image is its class.

Here is an exploratory visualization of the data set. It is two bar charts showing how the train data and test data are distributed in different classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because I tried with and without color information and gray image can make the validation result better.
Here is an example of a traffic sign image before and after grayscaling.

As a last step, I normalized the image data because for the nerual network likes the data to have zero mean and small range. I tried without the -.5~0.5 range normalization, the training process is not stable and have a low validation accuracy. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the eighth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn's method train_test_split

My final training set had 31367 number of images. My validation set and test set had 7842 and 12630 number of images.

I didn't augment the data set because I want to use another method to handle the imbalance issue. I tried to use class weight to justify the loss function but I didn't make it work.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x64
| RELU	    | |		
| Max pooling	    | 2x2 stride, outputs 6x6x64      			
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x64      			
| RELU	    |       		|
| Fully connected		| 160									|
| Fully connected  | 120 |
| Fully connected  | 43 |
| Softmax				| etc |
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth to twelfth cell of the ipython notebook. 

To train the model, I used an Adam optimizer, batch size 128, 20 epochs and 0.0008 as the learning rate.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.988
* test set accuracy of 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
    the default lenet architecture because it is easy to implement and fast to train also have a good performance
* What were some problems with the initial architecture?

        It works very well having training accuracy 0.98 and validation accuracy 0.95. I just think it was designed for digit recognition so the network is not big enough to describe the structures of traffic sign. So it is underfit.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


        1. I tried to twice all the depth of the lenet architecture, it works better than the original.
        2. I tried to add a third convolutional network without max pooling after it. It is much better and training accuracy and test accuracy are both higher
        3. I tried to add a dropout layer after the last conv network but it make the accuracy lower. Maybe my architecture is not overfit so generalization can't help. I should try a bigger network after I get a gpu server. Now I use my macbook pro to train the network and it is too slow.
* Which parameters were tuned? How were they adjusted and why?

        1. I changed the learning rate from 0.001 to 0.0008 because the loss of epochs in training swing. 
        2. I changed the epochs from 10 to 20 because the loss function is decreasing after 10 epochs.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

        1. I think cnn is good for this job because it can use the pixel position information in the image.
        2. I tried dropout layer, it did't work well. Maybe my network is not enough.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

        I tried a lot different architecture of the network and different parameters, but I don't know whether there is a good way to decide how to design the architecture, how many layers should we use and how many  nodes in each layer.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

All images except the fifth are difficult to classify because after resize to 32x32, the sign aspect are different from the training data.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| animals crossing   									| 
| Stop     			| Ahead only									|
| Vehicles over 3.5 metric tons prohibited					| No entry											|
| Speed limit (60km/h)	      		| Ahead only	 				|
| Ahead only			| Ahead only							|
| Keep right | No passing for vehicles over 3.5 metric tons |


The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 13.33%. This compares very low to the accuracy on the test set of 94.1%

The new images from the internet are very clear and don't have very complex background. I thought maybe the aspect of the sign changed because I resized the image to 32x32 and the original image is in different aspect ratio. So I cropped the image by hand and trained again in the 17th code cell, now the model can predict all signs correct.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h)    (probability of 1.0), and the image does contain a Speed limit (30km/h). The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)									| 
| 0     				| Speed limit (70km/h)									|
| 0				| Speed limit (20km/h)								|

For the second image, the model is relatively sure that this is a Stop    (probability of 1.0), and the image does contain a Stop. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop									| 
| 0     				| Wild animals crossing   									|
| 0				| Speed limit (50km/h) 

For the third image, the model is relatively sure that this is a Vehicles over 3.5 metric tons prohibited    (probability of 1.0), and the image does contain a Vehicles over 3.5 metric tons prohibited. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Vehicles over 3.5 metric tons prohibited									| 
| 0     				| Roundabout mandatory									|
| 0				| Priority road 

For the forth image, the model is relatively sure that this is a Speed limit (60km/h)
    (probability of 0.819), and the image does contain a Speed limit (60km/h)
. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.819         			| Speed limit (60km/h) | 
| 0.181     				| Speed limit (50km/h) |
| 0				| Speed limit (80km/h) | 

For the fifth image, the model is relatively sure that this is a Ahead only    (probability of 1.0), and the image does contain a Ahead only. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only									| 
| 0     				| Priority road									|
| 0				| Road work

For the sixth image, the model is relatively sure that this is a Keep right (probability of 1.0), and the image does contain a Keep right. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right									| 
| 0     				| Slippery road									|
| 0				| Turn left ahead    







