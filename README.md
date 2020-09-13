# Power_Lines_Detection
Prototype to detect insulators from UAV camera

Dataset: https://drive.google.com/drive/folders/1h4uj4hO9eyqr-qXLefe5TiEfVK0nAy3e?usp=sharing

Model's checkpoint: https://drive.google.com/file/d/1NEn0e2p_pGb0D6m9SbGwKTn4sgJUJocR/view?usp=sharing

Instructions for the dataset: Put Images and Annotations to the same folder and follow instructions 
in the notebook file

Model: ResNet-50 without pretrained layers saved as pickle object

So far model has been trained for 46 epochs

The code itlself is based on the this [guide](https://programmer.group/train-your-faster-rcnn-target-detection-model-using-pytorch.html) to organize data and 
Pytorch [libraries](https://github.com/pytorch/vision/tree/master/references/detection) for object detection.

Don't forget to change cv2_imshow inherited from Google Colab to plt.im_show during prediction 
