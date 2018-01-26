# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program


---
![cover_animation.gif animation](./results/cover_animation.gif)
---

In this semantic-segmentation project of term-3 of the self-driving car nanodegree program by Udacity, goal was to train a neural network model so that it labels the pixels of a road in provided test images using a Fully Convolutional Network (FCN).

## Project Goal & Notes

### Goal

The main goal of the project is that it labels or identifies most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. A solution that is acceptable, should label at least 80% of the road and label no more than 20% of non-road pixels as road.

#### Framework

The initial framework was provided by Udacity on github and can be found [here](https://github.com/udacity/CarND-Semantic-Segmentation)

To run this project, one should at least have the following packages installed (some other packages might be needed based on one's system configuration):

* [Python 3](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

#### Dataset and optional packages
My implementation focuses on Kitti Road dataset, to test on this, download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

I have also attempted segmentation on video-clip. For this some additional packages will be needed, like:

* [Moviepy](https://zulko.github.io/moviepy/)
* [Imageio](https://imageio.github.io/)


### Contents

* `main.py`: main code of the project which defines neural network architechture.
* `helper.py`: this script contains pre- and post-processing funstions, along with some attempted augmentation utilities.
* `project_tests.py`
* `main_video.py`: mostly same function as `main.py`. The only difference being that instead of training and saving trained model, as is done in `main.py` here the saved model is restored and applied to individual frames of a given video clip. 
* `data` - directory in which all test images or videos should be saved, like the `data_road` folder obtained from `Kitti Road dataset`. It is provided here just as a placeholder and hence is empty.
* `runs` - directory in which output images or videos will be saved. It is provided here just as a placeholder and hence is empty.
* `results` - directory that contains a few saved images from training, test or analysis performed during the project implementation.

#### Running the project

Though it is not necessary but use of GPU for the implementation of this project leads to significant time saving. I tried one run on both CPU and GPU with a very small data set, using 5 epochs with batch-size of 8. GPU took about 13 minutes while CPU took a little over 12 hours -:) .

* To run the implementation on any image run
```
python main.py
```

In the current form this will first download pre-trained VGG model to `data`, train the model, save the trained mode to a file and perform segmentation on test images. For use with alternate data set, it is recommeded that one verifies respective data-paths specified in `main.py`.

* To run the implementation on any video-clip run
```
python main_video.py
```

This will first read a pre-trained model saved to `./runs/sem_seg_model.ckpt` and perform segmentation on each frame of the video-clip, before producing the new segmented-video. Due to typically large size of pre-trained models, I have not provided any models here. Input and output video paths should be confirmed before running the script.


## Test & Implementation Details

Since this was one of the last projects, I used the opportunity to play around with data and hyperparameter selection in order to both improve my implementation (as much as possible in given time) and to enhance my understanding.

### Neural Network Architecture:

As suggested in the project walkthrough, a Fully Convolutional Network (FCN-8) has been used based on this [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). It uses VGG16 pretrained on ImageNet as an encoder. Decoder is used to upsample features, extracted by the VGG16 model, to the original image size. The decoder is based on transposed convolution layers.

### Hyperparameters

The final set of parameters I used were:

* `BATCH-SIZE` = 4
* `EPOCHS` = 50 (for static image segmentation)
* `EPOCHS` = 35 (for video-frame segmentation)
* `Learning rate` = 1e-4
* `Image augmentation` was limited to contrast & brightness modification

In the following sections I have tried to explain my logic or limitation behind selecting these parameters. To start with, here is a comparison between a sample original image and the image after road-pixel segmentation based on my implementation.

[image1]:./results/01_orig_um_000044.png
[image2]:./results/01_seg_um_000044.png

![alt text][image1]
![alt text][image2]

#### Batch Size selection

Lower number of batches means higher number of iterations in each EPOCH in neural network architechture, as far as I understand. However, most of the times the bottleneck is not how many iterations or forward/backward passes one can perform, but the computational power available, time-taken and size of final trained-model that is typically saved. Keeping all these in mind, I selected a `batch-size` of 4. Going lower to 2 would have probably improved my models performance further, but the time it was taking was too long and the trained model size was significantly higher.

To highlight this, in the following figure ([image3]) I have presented  a comparison between 4, 8 and 16 batch-sizes for a 20 epochs run. Clearly, a batch-size of 4 has lower losses for same number of epochs. This is also evident if we compare the segmented images with different batch-sizes, as is shown below in [image4].

[image3]:./results/plot_20_epoch_multiple_batch.jpg

[image4]:./results/batch_size_comparison.jpg

---

![alt text][image3]

![alt text][image4]

---

#### Epochs selection

Greater the number of Epochs means better the convergence, unless we reach saturation point. Hence, I kept increasing the number of Epochs till memory failure occured. I could manage to go up to 50 Epochs and 4 batch-size without any issue for static images. The mean-loss kept decreasing till this point, which is listed in the table below and is also highlighted in [image5] for a batch-size of 8. [image15] below just shows impact of number of epochs on actual segmentation on a sample image.

For image segmentation, I used Epoch=50, while for video-frame segmentation, I used Epochs=35.


| Epochs | Mean-loss | std-dev of loss |
|:---:|:---:|:---:|
|  5	 | 0.127630225 | 	0.028062905 |
| 10 | 	0.061038315 | 	0.01451998 |
| 20 | 	0.033932048 | 	0.007718682 |
| 30 | 	0.023161286 | 	0.006012847 |
| 50 | 	0.015464489 | 	0.003532481 |

[image5]:./results/plot_08_batch_multiple_epoch.png
[image15]:./results/epoch_comparison.jpg

![alt text][image5]
![alt text][image15]
---

#### Learning Rate

After playing around with learning rates between 1e-3 and 1e-5, the best one appeared to be 1e-4, hence I have used this throughout. The initial range was selected based on implementations in some past projects.

#### Image augmentation

Typically, image-augmentation improves trained model, hence, I tried augmentation based on simple brightness and contrast changes for randomly selected images, during the training process. This should have helped with reducing the impact of shadows and lighting conditions, I assume. Other image augmentation techniques like shearing, skew, mirroring, etc are possible to implement, however, I have not used them here.

For image brightness/contrast change I updated the image in `helper.py`, before it is passed for model training by adding the following section:

```
# randomly augment image
# change image contrast & brightness
act = random.randint(0,2)
# act==0 -- do nothing
if act==1: # brighten image
	image = image * random.uniform(1.1, 1.3) 
			+ random.randint(25, 100)
elif act==2: # darken image
	image = image * random.uniform(0.75, 0.95) 
			- random.randint(25, 100)

image[image>255] = 255
image[image<0] = 0

```

[image6]:./results/augmented_image_01.jpg
[image7]:./results/augmented_image_02.jpg
[image8]:./results/augmented_image_convergence.png

Two sample cases are shown in the [image6] and [image7] below.

---
![alt text][image6]
---
![alt text][image7]
---

On comparing the convergence performance of model trained on randomly-augmented with non-augmented images, we see that the augmented-image model converges a little faster [image8].

![alt text][image8]

#### Video Implementation

For video implementation, I use a pre-trained model with in-function image extraction, processing and masking. I arrived at this solution after a lot of reading online. Two sample videos can be found at the following locations:

* [video-1: stright-lanes](https://youtu.be/uv3psZL5UuI)
* [video-2: curved-lanes](https://youtu.be/1bljrJG--eA)

#### Sample Results

Some sample result-images using final parameter-set are shown below for reference.

[image9]:./results/um_000007.png
[image10]:./results/um_000015.png
[image11]:./results/umm_000008.png
[image12]:./results/umm_000069.png
[image13]:./results/uu_000070.png
[image14]:./results/uu_000093.png

---

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

---