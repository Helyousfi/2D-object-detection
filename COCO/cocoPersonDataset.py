################################
#  Create a dataset of person  #
################################
from pycocotools.coco import COCO
import os, io, cv2, sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib.patches as patches

# Load COCO Dataset
dataType='train2014'
annFile='annotations/instances_{}.json'.format(dataType)
coco=COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
print(len(imgIds))
show = 0

for k in range(len(imgIds)):
    img = coco.loadImgs(imgIds[k])[0]
    I = cv2.imread('train2014/'+img['file_name'], 0)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # Show the image
    if show:
        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray")
    H, W = I.shape
    # Set the following to 1 for img saving
    if 1:
        cv2.imwrite("person_dataset_bbox/images/coco_person_" + str(k) + ".png", I)
    for i in range(len(anns)):
        #print(anns[i]['bbox'])
        x = anns[i]['bbox'][0] / W 
        y = anns[i]['bbox'][1] / H 
        width = anns[i]['bbox'][2] / W
        height = anns[i]['bbox'][3] / H
        
        if show:
            image = I
            rect_pred = patches.Rectangle(  
                                (int(x * W), int(y * H)), 
                                int(width * W), 
                                int(height * H), 
                                linewidth = 1, 
                                edgecolor = [0, 0, 1, 1], 
                                facecolor = [0, 1, 0, 0.5])
            ax.add_patch(rect_pred)
    
        myData = [0, x, y, width, height]
        myFile = open(f'person_dataset_bbox/labels/coco_person_{str(k)}.txt', 'a', newline='')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)   
    if k % 100 == 0:
        print("Saved : ", k)
