# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_plot.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/center_2021_04_27_08_55_33_676.jpg "Center Image"
[image4]: ./examples/left_2021_04_27_08_55_33_676.jpg "Left Image"
[image5]: ./examples/right_2021_04_27_08_55_33_676.jpg "Right Image"
[image6]: ./examples/center_2021_04_27_08_55_33_676_normalize_minmax.jpg "Normalized to range"
[image7]: ./examples/center_2021_04_27_08_55_33_676_flip.jpg "Flipped Image"
[image8]: ./examples/center_2021_04_27_08_55_33_676_normalize_max.jpg "Normalized to max"
[image9]: ./examples/center_2021_04_27_08_55_33_676_normalize_sigma.jpg "Normalized to unit vector"
[image10]: ./examples/center_2021_04_27_08_55_33_676_crop.jpg "Cropped Image"
[image11]: ./examples/center_2021_04_27_13_18_05_411.jpg "Center Image Bridge"
[image12]: ./examples/left_2021_04_27_13_18_05_411.jpg "Left Image Bridge"
[image13]: ./examples/right_2021_04_27_13_18_05_411.jpg "Right Image Bridge"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **[model.py](model.py)** containing the script to create and train the model
* **[drive.py](drive.py)** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** summarizing the results
* **[video.mp4](video.mp4)** for recording of autonomous mode using model.h5 for 1 lap around Track1
* weights.h5 containing the weights of model.h5 in order to fine tune the model like transfer learning.
* [visualize.py](visualize.py) show summery and visualize graph of model.h5

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The code in [model.py](model.py) uses a Python generator to generate data for training rather than storing the training data in memory (code line 17 to 47 and line 109 to 113). The file shows the pipeline I used for training,validating, and saving the convolution neural network model, and it contains comments to explain how the code works.

Note: substitute the directory name of training data for `data/` in line 6 and line 30.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The neural network uses convolution layers with appropriate filter sizes of 5x5 and 3x3, and depths between 24 and 64 (model.py lines 63-103) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 65).

Two architecture were included: `LeNet-5` and [NVidia](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). The `LeNet-5` model reached at best about 0.026 for validation loss. The eventually used architecture is `NVidia` summarized as follow: 

|Layer (type)                 |Output Shape              |Param #   
|-----------------------------|--------------------------|-----------
|lambda_1 (Lambda)            |(None, 160, 320, 3)       |0         
|cropping2d_1 (Cropping2D)    |(None, 65, 320, 3)        |0         
|conv2d_1 (Conv2D)            |(None, 61, 316, 24)       |1824      
|max_pooling2d_1 (MaxPooling2 |(None, 30, 158, 24)       |0         
|conv2d_2 (Conv2D)            |(None, 26, 154, 36)       |21636     
|max_pooling2d_2 (MaxPooling2 |(None, 13, 77, 36)        |0         
|conv2d_3 (Conv2D)            |(None, 9, 73, 48)         |43248     
|conv2d_4 (Conv2D)            |(None, 7, 71, 64)         |27712     
|conv2d_5 (Conv2D)            |(None, 5, 69, 64)         |36928     
|flatten_1 (Flatten)          |(None, 22080)             |0         
|dense_1 (Dense)              |(None, 1164)              |25702284  
|dense_2 (Dense)              |(None, 100)               |116500    
|dense_3 (Dense)              |(None, 50)                |5050      
|dropout_1 (Dropout)          |(None, 50)                |0         
|dense_4 (Dense)              |(None, 1)                 |51        
|Total params: 25,955,233
|Trainable params: 25,955,233
|Non-trainable params: 0


#### 2. Attempts to reduce overfitting in the model
 
* Train/validation/test splits </br>
The model was trained and validated on different data sets to ensure that the model was not overfitting using `train_test_split()` of `sklearn.model_selection` (code line 13). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
* Attempts to reduce overfitting
  * Augmented data was used by flipping images form center, left and right camera.
  * Use cropped images to better focus on the road feature (model.py line 68).
  * The model contains dropout layers (model.py line 81 and line 102).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

The training data was recorded from center lane driving only, and kept the vehicle in the middle of lane as much as possible when recording. Since the vehicle was kept at center, the recovering steer when approaching lane edge was not recorded. Instead, samples taken by left and right camera were used to train recover steering by adding a constant parameter to get a larger steering angle.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a blank model of specific architecture with samples that captured from simulated track. Fine tune the model with new data which is recorded at the spots where the vehicle fell off the track previously.

My first step was to use a convolution neural network model similar to the `LeNet-5`. I thought this model might be appropriate because which is used for classifying traffic signs in the previous project and can recognize the edges and shapes of the signs and numbers from the first and second convolution layers, so it could probably sees the lanes on the roads.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At the beginning only the samples taken by center camera was used. And the validation loss is above 0.03. I trained the model with `data` provided by the workspace for 10 epochs. I found that my first model had a low mean squared error on the training set, and a similar low mean squared error on the validation set. But the validation error start to oscillate after the 5th epoch. This implied that the model was overfitting. 

To combat the overfitting, data augmentation and dropout layer were used in different stage.
|Step                 |Test Result              |Analysis   
|-----------------------------|--------------------------|-----------
| * Data augmentation </br> Flip images for 2x samples | Failed at the first left turn | Insufficient image sample at turning corner
| * Data augmentation </br> Add left and right camera images, including flip images,</br> total 6x samples than previous training. </br> * Steering corrective parameter </br> add 0.2 to steer back to center (code line 33 to 38)| Still failed at the first left turn | Corrective parameter may be too small
| Aggressive corrective parameter (0.3) | The validation loss was 0.286 and the model made it to the bridge


I modified the model to use `NVidia` architecture in the hope of passing the sharp left turn after the bridge.
|Step                 |Test Result              |Analysis   
|-----------------------------|--------------------------|-----------
| Modify model and trained with `data` in the workspace | Trained with 10 epochs and get 0.025 validation loss </br> fail at sharp left turn | overfitting
| * Add dropout layer (code line 81 and 102) | validation loss slowly reduced each epoch </br> but can't pass the sharp left turn | insufficient data
| * Fine tune the model weights </br> Record my own laps of training data **(1)**</br> in additional to `data` in the workspace | Trained for 5 epochs with validation loss less than 0.02</br> successfully passed the sharp left turn after the bridge, </br>but hit the bridge guardrail on the right side | Use the idea of transfer learning
| * Fine tune the model weights </br> Record of passing the bridge **(2)** | Trained for 5 epochs with validation loss less than 0.01 </br> kept in the center of the track

Each modification was tested by running the simulator to see how well the car was driving around track one. For the spots where the vehicle fell off the track, record new data and train the existed model to improve the driving behavior in these cases.
2 additional training data was recorded: (1) overall lap, and (2) passing the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-103) consisted of a convolution neural network which consists of 11 layers, including a normalization layer, a cropping layer, 5 convolutional layers (filter size is 5x5 for the first 3 layers and 3x3 for the last 2 layers), and 4 fully connected layers. 

Here is a visualization of the architecture, which can be generated using 
```sh
python visualize.py
```
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

As described above, 3 recoded data set was used. First trained with `data` in the workspace than saved the weights. Secondly, fine tuned with my own recorded lap. Finally, fine tuned with another recording of passing bridge. 

For the second part, I recorded 2 complete lap on Track1, one in clockwise and another in counter clockwise direction using center lane driving. To capture good driving behavior, I steered carefully with keyboard to keep the vehicle in the middle of the lane. Because the throttle is not considered in the test so the speed was as low as enough to keep the car moving. For me using the keyboard is easier to control the vehicle. After finishing the first lap, I paused and made a U turn then recorded the second lap in counter clockwise direction.

During center lane driving, left side and right sides image also recorded in the meantime. These images has a corresponding throttle value. As described above, adding the throttle by 0.3 for the left image to make the vehicle turn right. In the other hand, reducing the throttle by 0.3 for the right image to make the vehicle turn left.  Thus the model learned how to recover back to the track.

Here are left, center and right image taken by 3 camera: 

![from left camera][image4]
![from center camera][image3]
![from right camera][image5]

I was going to do the same thing for Track2 but kept falling from cliff, or rolling back on a hill. The auto throttle is not enough for uphill. So far only Track1 was recorded.

To augment the data sat, I also flipped images and angles thinking that this would helps balancing the left and right turn scenes. For example, left image below is the original image taken by center camera and the right one has then been flipped:

![origin][image3]
![flip][image7]

After the collection process, there are 10940 x 3 x 2 images in the second training and validation samples.
I then preprocessed this data by setting up lambda layer to normalized and mean_centered the input pixels (code line 65). The image has been linear mapping from [0,255] to [-0.5,0.5] without changing its contrast but it is crucial for model training. Other types of normalization is not used in training but would be helpful if our sample includes varying lighting condition. Using [cv::normalize()](https://docs.opencv.org/master/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd), normalize the min and max value to range [0,255] will enhance the contrast, normalize to the max element do similar, and normalize to unit vector. The normalized results were processed in order to show as image. 
![alt text][image6]
![alt text][image8]
![alt text][image9]

I finally randomly shuffled the data set and put 20% of the data into a validation set.
The image samples were than cropped to better focus on the road feature.
![alt text][image10]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by experiment that shows the validation loss decreased vary slow after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

After the first (trained by `data` of workspace) and second (trained by my records) of training, the model runs well except at the bridge. So the new records of passing bridge was taken to fine-tune the model.
![alt text][image12]
![alt text][image11]
![alt text][image13]

And the [video.mp4](video.mp4) shows the result of this model.
