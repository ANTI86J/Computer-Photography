# Computer-Photography-project1
CSCI6907 in GWU with Dr.pless

1. edge detection
I used four ways of edge detection.We notice the the edge of this image is a white line and a black line, so I have couple of ideas and made it work in four ways. First we set a threshold, if the pixel is above the threshold, then it would turn to 255, so that we used the function in python opencv called cv2.threshold which can help me to do this.After implementing this, we can get a blank and white picture with a black and white line around the image, and then we just need to extract those lines. Second way is that we travel through all the image, but considering that some images would be very large, so we can first scale down the size of this picture in the same height-and-width propotion, and then we set a threshold t which is the sum of very column/row and is no bigger than t * row/column, we record very row/column, so that we find the upper and lower limit of the real image.Third method is use the function 'edge' which is in the Matlab.In the function edge, we use sober method to finish the job.And when we travel through the image, we use edge and function 'bwareaopen' to get the line of the edge.The fourth way is that we travel through the image, calculate the similarity of every two rows/columns, find the first mutations in value(if we do this forward and backward), but this has some limitations for some images also have large margins of white space(like describing the sky or maybe a full view of a village), so i haven't included it in my image, but it works!
(The link of the code is in the end of this write-up)

2. image warp function
As we can see, we can use transformer.warp function to apply image homography, but 
if we want to use this, we must extract the points of interest, so in order to do this, I use Google Vision API to extract the points of face as the points of interest, and then I just used this function to change the angle of the face to get a better view.
Another time I use image warp function is in the transformation of different color spaces, which I leave it in the next sections as belows.


3.transform between different color spaces
I used three ways to do this, first is a simple scale implentation, just for every channel of this image, just extract every 16 * 16 matrix, and then just compare the similarity of those matrix in the same position(row) of different color channel, then find the most aligned position of each three channels, and then adjust them to be aligned, and finally use cv2.merge to merge those three color channel. Second method is called ECC algorithm which used to align the images according to the "Parametric Image Alignment using Enhanced Correlation Coefficient Maximization"(link:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4515873).The reason I used this method is that although there might not be much correlation in greyscale RGB space, but we can notice that in gradient domain, things can be different, and it has strong adaptability to image contrast and brightness changes and simple iterative solution although the objective function is nonlinear! First we need to calculate the gradient of the image in axis x and y, at the same time we created a eye matrix which used to be the initial warp matrix, then based on the theory I mentioned above, we can use this function in opencv to get the warpmatrix, and change every color channel of this image in the warp perspective to get the aligned color-space, then just merge these! The third way is in the Matlab aiming to get the best aligned direction of axis x and y

briefly, it searches for the most similar pixel in the surronding.


4.bells and Whistles
1) about edge detection, I have detected the edge of the image which has been written in the edge detection part
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

![00458u](https://user-images.githubusercontent.com/34802668/153046620-d9044c72-c3d7-4bfd-a8e1-329e886166ff.jpg)
![01598v](https://user-images.githubusercontent.com/34802668/153046657-29f58c5b-7888-4961-81ea-a9b94b8eae39.jpg)
![31421v](https://user-images.githubusercontent.com/34802668/153046697-2cd45ccf-34ad-43af-926c-00d76b03df5c.jpg)
![00163v](https://user-images.githubusercontent.com/34802668/153046845-b1146d76-69f0-495b-a3e9-ea81479cfb93.jpg)


