import cv2

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import copy



def NMS(boxes, overlapThresh = 0.4):
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > treshold:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

def bounding_boxes(image, template):
    img_array = image.transpose(1, 2, 0)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Create a selective search object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Set the input image
    ss.setBaseImage(img_cv2)

    # Perform selective search
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    # Return the bounding boxes
    return rects

def draw_bounding_boxes(image,boxes):
    for box in boxes:
        image = cv2.rectangle(copy.deepcopy(image),box[:2], box[2:], (255,0,0), 3)
    return image

if __name__ == "__main__":
    time.sleep(2)
    treshold = 0.8837 # the correlation treshold, in order for an object to be recognised
    template_diamonds = plt.imread(r'000032.jpg')

    ace_diamonds_rotated = plt.imread(r'000032.jpg')

    boxes_redundant = bounding_boxes(ace_diamonds_rotated, template_diamonds) # calculate bounding boxes
    boxes = NMS(boxes_redundant)                                            # remove redundant bounding boxes
    overlapping_BB_image = draw_bounding_boxes(ace_diamonds_rotated,
                                               boxes_redundant)  # draw image with all redundant bounding boxes
    segmented_image = draw_bounding_boxes(ace_diamonds_rotated,boxes)           # draw the bounding boxes onto the image
    plt.imshow(overlapping_BB_image)
    plt.show()
    plt.imshow(segmented_image)
    plt.show()
