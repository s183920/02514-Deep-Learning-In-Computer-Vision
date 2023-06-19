import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def selective_search(img):
    """Performs selective search on the input image (tensor)."""
    print("Running selective search..")
    # Convert image to cv2
    img_array = img.cpu().numpy().transpose(1, 2, 0)
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


def NMS(rect_list, threshold = 0.4):
    """
        Non maximum supression (NMS) to remove overlapping bounding boxes. 
        Default behavior is 40% of the boxes is allowed to be overlapping. Else other boxes are removed. 
    """
    print("Running Non-Maximum Supression!")
    if len(rect_list) == 0:
        return []
    print("Original no of bboxes: ", len(rect_list))
    x1 = rect_list[:, 0]  # x coordinate of the top-left corner
    y1 = rect_list[:, 1]  # y coordinate of the top-left corner
    x2 = rect_list[:, 2]  # x coordinate of the bottom-right corner
    y2 = rect_list[:, 3]  # y coordinate of the bottom-right corner

    # Compute area.
    areas = (x2-x1 + 1) * (y2-y1 +1) # Adding +1 to pad for border pixels. 

    indices = np.arange(len(x1))

    for i, box in enumerate(rect_list):
        temp_indices = indices[indices != i]
        xx1 = np.maximum(box[0], rect_list[temp_indices,0])
        yy1 = np.maximum(box[1], rect_list[temp_indices,1])
        xx2 = np.minimum(box[2], rect_list[temp_indices,2])
        yy2 = np.minimum(box[3], rect_list[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        #print("overlap is: ",overlap)
    
    indices = [i for i, num in enumerate(overlap) if num > threshold]
    #print("Theeese exceed the threshold: ", indices)
    print("Final no of boxes after NMS", len(indices))
        
    return rect_list[indices].astype(int)
    


def drawReduced(image, rect_list):
    image_copy = image.copy()
    if image_copy.dtype == "float32":
        image_copy *= 255


    for box in rect_list:
        cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    return image_rgb.astype("uint8")
def draw_rectangles(image, rects, annotation):
    rect_list = []
    # Create a copy of the original image
    image_copy = image.copy()
    if image_copy.dtype == "float32":
        image_copy *= 255
   
    # Draw rectangles on the image
    for r in rects:
        x1, y1, x2, y2 = rect_coordinates(r, annotation)
        rect_list.append((x1,y1,x2,y2))
        array = np.array(rect_list)
        # # x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Convert the image from BGR to RGB
    print("appended coordinates to np.Array")
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    


    return image_rgb.astype("uint8"), array

def rect_coordinates(rect, annotation):
    #print("Converting coordinates")
    """Converts the coordinates of a rectangle from selective search to the coordinates used in the annotations."""
    x, y, w, h = rect
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    x1 /= annotation["width_scale"]
    y1 /= annotation["height_scale"]
    x2 /= annotation["width_scale"]
    y2 /= annotation["height_scale"]

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return x1, y1, x2, y2

if __name__ == "__main__":
    from data import TacoDataset, get_dataloader, show_img

    # data
    dataset = TacoDataset(datatype="train", img_size=(800, 800))
    data_loader = get_dataloader(dataset)

    # perform selective search
    for imgs, annotations in data_loader:
        for img, annotation in zip(imgs, annotations):
            # run selective search
            results = selective_search(img)
            # show results
            fig, ax = plt.subplots()
            img_orgBox, array = draw_rectangles(img.cpu().numpy().transpose(1, 2, 0), results, annotation)
            ax.imshow(img_orgBox)
            ax.set_axis_off()
            plt.savefig("OriginalBBOX.png")
            ReducedBB = NMS(array, threshold=0.3)
            reducedImg = drawReduced(img.cpu().numpy().transpose(1,2,0), ReducedBB)
            ax.imshow(reducedImg)
            ax.set_axis_off()
            plt.savefig("NMSBBOX.png")
            break
            
        break
