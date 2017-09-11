## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: output_images/car_not_car.png
[image2]: output_images/histogram_of_color.png
[image3]: output_images/spatial_binning.png
[image4]: output_images/HOG_example.png
[image5]: output_images/normalized.png
[image6]: output_images/search_windows.png
[image7]: output_images/detected_windows.png
[image8]: output_images/windows1.png
[image9]: output_images/windows2.png
[image10]: output_images/windows3.png
[image11]: output_images/final_windows1.png
[image12]: output_images/final_windows2.png
[image13]: output_images/final_windows3.png
[image14]: output_images/heatmap1.png
[image15]: output_images/heatmap2.png
[image16]: output_images/heatmap3.png
[image17]: output_images/heatmap4.png
[image18]: output_images/windows0.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Most of the functions used in this project are defined in the 3rd cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Before using HOG features, I extracted the histogram of color and also spatial binning, this is tested in cells 4 and 5 respectively, and here is an image of the results for both of this features.
![alt text][image2]
![alt text][image3]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

After testing several parameters, I choose the following

| Feature       | Parameter      | Values   |
|:-------------:|:--------------:|:-------------:|
| All           | color space    | YCrCb
| HOG           | orientation    | 9
| HOG           | pix per cell   | 8
| HOG           | cell per block | 2
| HOG           | channels       | All
| Color Hist    | bins           | 32
| Color Hist    | range          | (0,256)
| Spatial Bin   | size           | (16,16)

All the values are joined together and normalized, producing the following results

![alt text][image5]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After selecting the best parameters I train a SVM in cells 8 and 9 of the notebook.

At first I was using a non-linear kernel but this was very slow, so I went back to a linear one that works faster but still gives good results.

With this parameters my feature vector has a length of 6156 and the classifier has an accuracy of 99%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The first approach for sliding windows consists of first getting the location for all the windows that will be scanned, once we have this windows we extract features for each and compare it with our model to see if the window has a car

![alt text][image6]
![alt text][image7]

A better approach is to extract the HOG features only once for each image and then apply the sliding windows over those to see if there is a match, this is tested in cell #11

Then in cell 12 I use a function to search multiple sizes and retrieve the results I decided to use 3 different scales because if I added more the processing time is too slow

scale 1
![alt text][image18]

scale 1.5
![alt text][image8]

scale 2.5
![alt text][image9]


With the combined windows I get the following final result

![alt text][image10]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (1.5 and 2.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image11]
![alt text][image12]
![alt text][image13]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/3GXfiMyczAg)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code does a good job detecting the cars, it struggles
with the cars farther away, or to the sides, this could be improved by adding more scales for the windows search, however this will increase the already slow processing time.

Also the heatmap detection can be improved, I was thinking to add heat to the whole window if there was an overlap, I was not able to test this due to time constraints.

Overall the processing times needs to be optimized to work faster if this wants to be used in real time
