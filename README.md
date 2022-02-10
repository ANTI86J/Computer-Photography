# Computer-Photography-project1
CSCI6907 in GWU with Dr.pless

1. edge detection
I used four ways of edge detection.We notice the the edge of this image is a white line and a black line, so I have couple of ideas and made it work in four ways. First we set a threshold, if the pixel is above the threshold, then it would turn to 255, so that we used the function in python opencv called cv2.threshold which can help me to do this.After implementing this, we can get a blank and white picture with a black and white line around the image, and then we just need to extract those lines. Second way is that we travel through all the image, but considering that some images would be very large, so we can first scale down the size of this picture in the same height-and-width propotion, and then we set a threshold t which is the sum of very column/row and is no bigger than t * row/column, we record very row/column, so that we find the upper and lower limit of the real image.Third method is use the function 'edge' which is in the Matlab.In the function edge, we use sober method to finish the job.And when we travel through the image, we use edge and function 'bwareaopen' to get the line of the edge.The fourth way is that we travel through the image, calculate the similarity of every two rows/columns, find the first mutations in value(if we do this forward and backward), but this has some limitations for some images also have large margins of white space(like describing the sky or maybe a full view of a village), so i haven't included it in my image, but it works!
(The link of the code is in the end of this write-up).
About what I think, at first I just thought using some specific patterns about the black and white edge, but after run some samples, I find that the edge is not purely white and black, so the first thing in my mind is that, the black and white are the extreme point in the whole line, so it's must far beyond some average points, so I just discover a threshold to do this job, and I just calculate the max columns in the left and the min colomns in the right, both number means the edge of this img, and it turns out to work very well, here are some examples.So,after I found that this could work, I just think of a function in cv2 called cv2.threshold, and it works well too!

![00056v](https://user-images.githubusercontent.com/34802668/153486860-e66c5540-a804-4882-94a9-99a75f70c95b.jpg)


2. image warp function
As we can see, we can use transformer.warp function to apply image homography, but 
if we want to use this, we must extract the points of interest, so in order to do this, I use Google Vision API to extract the points of person as the points of interest, and then I just used this function to change the angle of the face to get a better view.Just like the bigger green bounds

<img width="374" alt="image" src="https://user-images.githubusercontent.com/34802668/153487680-890f0e6b-a7c2-4661-87b2-50903a0d6d07.png">

, and then I got those normalized coordinates:

<img width="374" alt="image" src="https://user-images.githubusercontent.com/34802668/153488456-e52289e0-8935-4829-a181-9ae193cb956c.png">

and next I just need to use this coordinate to do the warping!In order to get a ampilied image, I made this woman a bit more fatterm, so that I could amplify some details more clearly

<img width="550" alt="image" src="https://user-images.githubusercontent.com/34802668/153491486-b331642a-cdb0-44ee-a689-4a60b59a35c1.png">

it is just one of the projections I made to get a bigger view of the woman so I must stand some loss of the clear of the img, but in some other cases, I wouldn't do that, just show how I think and adjust that, for different cases, I use different projections


Another time I use image warp function is in the transformation of different color spaces, which I leave it in the next sections as belows.


3.transform between different color spaces
I used three ways to do this, first is a simple scale implentation, just for every channel of this image, just extract every 16 * 16 matrix, and then just compare the similarity of those matrix in the same position(row) of different color channel, then find the most aligned position of each three channels, and then adjust them to be aligned, and finally use cv2.merge to merge those three color channel. Second method is called ECC algorithm which used to align the images according to the "Parametric Image Alignment using Enhanced Correlation Coefficient Maximization"(link:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4515873)
.The reason I used this method is that although there might not be much correlation in greyscale RGB space, but we can notice that in gradient domain, things can be different, and it has strong adaptability to image contrast and brightness changes and simple iterative solution although the objective function is nonlinear! First we need to calculate the gradient of the image in axis x and y, at the same time we created a eye matrix which used to be the initial warp matrix, then based on the theory I mentioned above, we can use this function in opencv to get the warpmatrix, and change every color channel of this image in the warp perspective to get the aligned color-space, then just merge these! The third way is in the Matlab aiming to get the best aligned direction of axis x and y

briefly, it searches for the most similar pixel in the surronding.
Just like the img and img_transformed, I just use this method to transform from BGR to RGB
This is the BGR view:

<img width="550" alt="image" src="https://user-images.githubusercontent.com/34802668/153492051-39d07b4b-396e-481e-92ab-a651ac243cb4.png">

This is the RGB view:

<img width="550" alt="image" src="https://user-images.githubusercontent.com/34802668/153491486-b331642a-cdb0-44ee-a689-4a60b59a35c1.png">

4.Multi-scale aligning algorithm 
At first, I kinda don't understand the meaning of multi-scale aligning algorithm, after I finished the single-scale aligining, I just try to google what and how I can do to make it, then I find this one and use it.The key of the multi-scale aligning algorithm is a simple 3×3 matrix called Homography.And I checked and read the wikipedia related to this.(link:https://en.wikipedia.org/wiki/Homography),and I finally get that, but the math here needs some related knowledge, so I just made it more simple to explain how I do it.
Homography is that like that:

<img width="256" alt="image" src="https://user-images.githubusercontent.com/34802668/153493746-8e646875-88f2-4834-abaf-09cda4b03e9d.png">

(x1, y1) is is the point in the first image in one color-space, (x2, y2) is the coordinates of the second image at the same physical point. Homography then correlates them in the following way

<img width="533" alt="image" src="https://user-images.githubusercontent.com/34802668/153493801-06a39095-6247-4993-b503-59e7cd13b72e.png">

and then I use the the findHomography function in cv2 to get this goal! Just like that(some tests on different datasets):

<img width="708" alt="image" src="https://user-images.githubusercontent.com/34802668/153494439-8107dac2-b5a2-46b0-904c-98048560ead7.png">


So I wonder how I can get those points automatically, first i tried to use gradient, cause no matter how this image change, the gradient of some certain pattern does not change, but it takes so much time to do it, so I try to find whether opencv could help me do that, luckily,
Multiple keypoint detectors (such as SIFT, SURF, and ORB) are implemented in OpenCV, in my codes I used ORB. Just like that(some tests on different datasets):

<img width="708" alt="image" src="https://user-images.githubusercontent.com/34802668/153494580-3f9eda06-d5d2-479f-abb3-3c264f121d07.png">

But after I run some samples in the data, I just found that the imgs in different color space are not perfectly matched with each other. And I carefully read the official file of findHomography, and fortunately, the findHomography method utilizes a robust estimation technique called RANSAC, or random sample consistency, that produces the right results even in the presence of a large number of bad matches.Once the correct homography is calculated, the transformation can be applied to all pixels in one image to map them to another. This is done using the warpPerspective function in OpenCV.
So here is the final result of this multi-scale aligning algorithm(just list one of those):

![01269v](https://user-images.githubusercontent.com/34802668/153495277-91996e6b-4f2d-4d58-b823-92521ea6c17f.jpg)


5.bells and Whistles
1) about edge detection, I have detected the edge of the image which has been written in the edge detection part(which used a threhold to detect the real edge of one img, and then just crop it), here is the result:

![00125v](https://user-images.githubusercontent.com/34802668/153486955-ba986250-445c-4ba1-a296-5ac7ef3ab8b0.jpg)

2) to enhance contrast,I used the Histogram normalization.Formulation is showed below
calculate formulation
when the dtype is cv2.NORM_MINMAX, the formulation is showed below

3) better features.I used the gradients which has been written in the transformation part

4) better colors: We can transform the color channel from BGR space to HSV or Ycrcb space.And sometimes the color in the image is needed to be enhanced so I wrote 5 algorithms about white-balanced which is shown in the code.

5) better transformations.Just wrote in the transformation part,it can be shown in my code

Some results:

from matlab:


![result-01269v](https://user-images.githubusercontent.com/34802668/153045696-db248761-e93d-4277-9420-38208258613e.jpg)
![result-01597v](https://user-images.githubusercontent.com/34802668/153045783-71a31ba9-e62f-428f-a09e-a947950af539.jpg)
![result-10131v](https://user-images.githubusercontent.com/34802668/153045851-5d448782-a40a-46ee-b4aa-88a8acee4787.jpg)

from python:

![00458u](https://user-images.githubusercontent.com/34802668/153047499-d71277b7-d139-44a5-9a6a-bdbdf37c5c3a.jpg)
![01598v](https://user-images.githubusercontent.com/34802668/153047671-23791e7a-6218-4381-9c1c-3c7ed71121d6.jpg)
![31421v](https://user-images.githubusercontent.com/34802668/153047815-b2f41e32-177e-4689-88ea-b15ab7041787.jpg)
![01164v](https://user-images.githubusercontent.com/34802668/153048172-69230664-fb59-49bd-b4f9-89b59ce6c7eb.jpg)


