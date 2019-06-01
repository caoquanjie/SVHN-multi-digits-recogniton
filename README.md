# SVHN-multi-digits recogniton
a tensorflow version implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082).

## SVHN dataset
SVHN is obtained from house numbers in Google Street View images. The dataset can be download [here(format1)](http://ufldl.stanford.edu/housenumbers/).</br>

## Training data samples
![](https://github.com/caoquanjie/SVHN-multi-digits-recogniton/raw/master/images/sample.jpg)

## Requirements
python 3.6</br>
tensorflow 1.4.0</br>
numpy 1.15.0</br>
matplotlib 2.0.0</br>

## Training details
we generate images with bounding boxes, and resize the images to 64×64. 
We then use the similar data augmentation which crops a 54×54 pixel image from a random location within the 64×64 pixel image in [Goodfellow et al. (2013)](https://arxiv.org/pdf/1312.6082).</br>


Run `python convert_to_tfrecords.py`, you can get three tfrecords files.</br>
Run `python main.py`

## Graph
![](https://github.com/caoquanjie/SVHN-multi-digits-recogniton/raw/master/images/tensorboard_graph.png)

## Result
The recognition accuracy of this model is reached 96.04%.
All qualitative and quantitative results are all exported to the svhn.log, you can print some other results to the logs if you are interested.
You also can view results in tensorboard.</br>



Run `tensorboard --logdir=logs`



