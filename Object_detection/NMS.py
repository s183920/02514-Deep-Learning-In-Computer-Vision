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