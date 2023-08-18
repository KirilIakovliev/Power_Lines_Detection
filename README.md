
IT IS FOR JENKINS


NEW LINES 


# Power_Lines_Detection
Prototype to detect insulators from UAV camera

Dataset: https://drive.google.com/drive/folders/1h4uj4hO9eyqr-qXLefe5TiEfVK0nAy3e?usp=sharing

Model's checkpoint: https://drive.google.com/file/d/1NEn0e2p_pGb0D6m9SbGwKTn4sgJUJocR/view?usp=sharing

Instructions for a new training: Put Images and Annotations to the same folder and follow instructions 
in the notebook file

To make predictions just upload the model's checkpoint and put it in the same folder with .ipynb file.
To make predictions go to the Prediction part 

Model: ResNet-50 without pretrained layers. Saved as a pickle object

So far model has been trained for 46 epochs

The code itlself is based on this [guide](https://programmer.group/train-your-faster-rcnn-target-detection-model-using-pytorch.html) to organize data and 
Pytorch [libraries](https://github.com/pytorch/vision/tree/master/references/detection) for object detection.

![Prediction](examples/prediction.jpeg?raw=true "Title")
