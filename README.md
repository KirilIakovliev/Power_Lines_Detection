# Power_Lines_Detection
Prototype to detect insulators from UAV camera

Dataset: https://drive.google.com/drive/folders/1h4uj4hO9eyqr-qXLefe5TiEfVK0nAy3e?usp=sharing

Model: ResNet-50 without pretrained layers

So far model has been trained for 40 epochs

The code itlself is based on the this [guide](https://programmer.group/train-your-faster-rcnn-target-detection-model-using-pytorch.html) to organize data and 
pytorch [libraries](https://github.com/pytorch/vision/tree/master/references/detection) for object detection.

Don't forget to change cv2_imshow inherited from Google Colab to plt.im_show during prediction 
