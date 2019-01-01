# Semantic Segmentation

![sample img](./examples/sample.png) 
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Training

Run the following command to run the project:
```
python main.py epochs batch_size
```

#### Data augumentation

In this project, I flip the image along y axis to enrich the data. 

 |Original Image         |  Augumented Image
:-------------------------:|:-------------------------:
![original img](./examples/img.png)  |  ![flipped img](./examples/img_flip.png)
![original img](./examples/gt.png)  |  ![flipped img](./examples/gt_flip.png)

#### Loss visualization

![training loss](./examples/training_loss.png)