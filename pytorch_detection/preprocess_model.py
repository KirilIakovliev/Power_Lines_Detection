
import os
import torch
import numpy as np
import transforms as T
from PIL import Image
from xml.dom.minidom import parse

class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")        
        
        # Read file, VOC format dataset label is xml format file
        dom = parse(bbox_xml_path)
        # Get Document Element Object
        data = dom.documentElement
        # Get objects
        objects = data.getElementsByTagName('object')        
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # Get the contents of the label
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # Is label, insulator or defect
            if name == 'norm_insulator':
              labels.append(np.int(1.0))  # insulator and defect labels are 0 and 1, respectively
            elif name == 'defect_insulator':
              labels.append(np.int(0.0))
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Since you are training a target detection network, there is no target [masks] = masks in the tutorial
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            # Note that target (including bbox) is also transformed\enhanced here, which is different from transforms from torchvision import
            # Https://github.com/pytorch/vision/tree/master/references/detectionOfTransforms.pyThere are examples of target transformations when RandomHorizontalFlip
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50% chance of flipping horizontally
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)