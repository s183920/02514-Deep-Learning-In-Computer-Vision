import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision
import torch


def selective_search(img, max_proposals):
    """Performs selective search on the input image (tensor) and limits the number of proposals."""
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

    # Limit the number of proposals
    if len(rects) > max_proposals:
        rects = rects[:max_proposals]

    # Return the bounding boxes
    return rects


def non_maximum_suppression(boxes, scores, threshold):
    """Applies non-maximum suppression to the bounding boxes based on their scores and IoU threshold."""
    # Sort boxes by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]

    
    # Initialize a list to store the selected box indices
    selected_indices = []
    
    # Iterate over the sorted boxes
    for i in range(len(boxes)):
        box = boxes[i]
        
        # Calculate IoU with previously selected boxes
        # ious = calculate_iou(box, boxes[selected_indices])
        b1 = torch.from_numpy(box.reshape(1, *box.shape))
        b2 = torch.from_numpy(boxes[selected_indices])
        ious = torchvision.ops.box_iou(b1, b2)
        # print(ious)

        # Check for NaN values in ious array
        if np.isnan(ious).any():
            continue
        
        # Get indices of boxes with IoU less than the threshold
        overlapping_indices = np.where(ious > threshold)[0]
        
        # Check if the box overlaps with any previously selected boxes
        if len(overlapping_indices) == 0:
            selected_indices.append(i)
    
    # Return the selected boxes
    return boxes[selected_indices], selected_indices


def calculate_iou(box, boxes):
    """Calculates the Intersection over Union (IoU) between a box and a list of boxes."""
    # Calculate intersection area
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    # Calculate union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    
    # Calculate IoU
    # print(intersection_area, union_area)
    iou = intersection_area / union_area
    
    return iou




def draw_rectangles(image, rects, annotation):
    # Create a copy of the original image
    image_copy = image.copy()
    if image_copy.dtype == "float32":
        image_copy *= 255

    # Draw rectangles on the image
    for r in rects:
        x1, y1, x2, y2 = rect_coordinates(r, annotation)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    return image_rgb.astype("uint8")


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
    import torch
    from model import Resnet
    # data
    dataset = TacoDataset(datatype="train", img_size=(800, 800))
    data_loader = get_dataloader(dataset)
    model = Resnet() 
    scores = []
   
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    # perform selective search
    for imgs, annotations in data_loader:
        for img, annotation in zip(imgs, annotations):
            # run selective search
            results = selective_search(img,max_proposals=100)

            # Convert results to array format
            boxes = np.array(results)
            model.eval()
            
            # Iterate over boxes..
            for box in boxes: 
                xmin, ymin, w, h = box
                xmax = xmin + w
                ymax = ymin + h
                print(xmax - xmin, ymax - ymin) 
                if (xmax - xmin) < 7 or (ymax - ymin) < 7:
                    print(xmax - xmin, ymax - ymin)
                    continue
                xmin /= annotation["width_scale"]
                xmin = int(xmin)
                xmax /= annotation["width_scale"]
                xmax = int(xmax)
                ymin /= annotation["height_scale"]
                ymin = int(ymin)
                ymax /= annotation["height_scale"]
                ymax = int(ymax)


                proposal_image = img[:, xmin:xmax, ymin:ymax]
                if proposal_image.shape[1] == 0 or proposal_image.shape[2] == 0:
                    continue
                proposal_image = proposal_image.unsqueeze(0)
                score = model(proposal_image)
                class_conf  = torch.max(score)
                # append to an array of unknown size
                scores = np.append(scores, class_conf.detach().numpy())


            # Apply non-maximum suppression
            threshold = 0.7 # Example threshold value
            reduced_boxes = non_maximum_suppression(boxes, scores, threshold)

            # Show results
            fig, ax = plt.subplots()
            img_orgBox = draw_rectangles(
                img.cpu().numpy().transpose(1, 2, 0), results, annotation
            )
            ax.imshow(img_orgBox)
            ax.set_axis_off()
            plt.savefig("OriginalBBOX.png")

            reducedImg = draw_rectangles(
                img.cpu().numpy().transpose(1, 2, 0), reduced_boxes, annotation
            )
            ax.imshow(reducedImg)
            ax.set_axis_off()
            plt.savefig("NMSBBOX.png")

            break

        break
