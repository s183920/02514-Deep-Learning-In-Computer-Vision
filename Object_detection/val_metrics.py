# import torch

# def IoU(prediction, ground_truth):
#     intersection = torch.max(torch.min(prediction[:, 0], ground_truth[:, 0]) * torch.min(prediction[:, 1], ground_truth[:, 1]), torch.zeros(ground_truth.shape[0]))
#     union = prediction[:, 0] * prediction[:, 1] + ground_truth[:, 0] * ground_truth[:, 1] - intersection
#     return intersection / union

# def average_precision(confidence_scores, bounding_boxes, ground_truth):
#     '''
#     Average Precission
    
#     input: 
#         confidence_scores: confidence scores of prediction
#         bounding_boxes: bounding boxes of prediction
#         ground_truth: ground truth

#     output:
#         AP: Average Precission
#     '''
#     # get prediction
#     prediction = torch.cat((confidence_scores.unsqueeze(1), bounding_boxes), dim=1)
#     # sort prediction by confidence score
#     prediction = prediction[prediction[:, 0].argsort(descending=True)]
#     # get ground truth
#     ground_truth = ground_truth[ground_truth[:, 0].argsort(descending=True)]
#     # get number of ground truth
#     num_ground_truth = ground_truth.shape[0]
#     # get number of prediction
#     num_prediction = prediction.shape[0]
#     # get number of classes

#     # get true positive
#     true_positive = torch.zeros(num_prediction)
#     for i in range(num_prediction):
#         # get prediction
#         prediction_i = prediction[i]
#         # get ground truth
#         ground_truth_i = ground_truth[ground_truth[:, 1] == prediction_i[1]]
#         # get iou
#         iou = IoU(prediction_i[2:], ground_truth_i[:, 2:])
#         # get max iou
#         max_iou, max_iou_index = torch.max(iou, dim=0)
#         # get max iou index
#         max_iou_index = max_iou_index[max_iou > 0.5]
#         # if max iou index is not empty
#         if max_iou_index.shape[0] > 0:
#             # get max iou index
#             max_iou_index = max_iou_index[0]
#             # if max iou index is not used
#             if ground_truth_i[max_iou_index, 0] == 0:
#                 # set true positive to 1
#                 true_positive[i] = 1
#                 # set ground truth to used
#                 ground_truth_i[max_iou_index, 0] = 1

#     # get cumulative sum of true positive
#     true_positive_cumsum = torch.cumsum(true_positive, dim=0)
#     # get cumulative sum of prediction
#     prediction_cumsum = torch.cumsum(torch.ones(num_prediction), dim=0)
#     # get precision
#     precision = true_positive_cumsum / prediction_cumsum
#     # get recall
#     recall = true_positive_cumsum / num_ground_truth
#     # get average precission
#     AP = torch.sum(precision * true_positive) / num_ground_truth

#     return AP, precision, recall


import numpy as np

def calculate_iou(bbox1, bbox2):
    # Calculate the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    # Calculate the IoU
    iou = intersection / union
    return iou

def calculate_average_precision(ground_truth, boxes, confidence_scores, iou_threshold=0.5):
    ground_truth = np.concatenate((ground_truth.numpy(), np.zeros((len(ground_truth), 1))), axis=1)
    detections = np.concatenate((boxes.numpy(), confidence_scores.unsqueeze(1).numpy()), axis=1)
    num_detections = len(detections)
    true_positives = np.zeros(num_detections)
    false_positives = np.zeros(num_detections)

    # Sort the detections by confidence score (highest to lowest)
    sorted_indices = np.argsort(detections[:, 4])[::-1]
    detections = detections[sorted_indices]

    # Initialize a list to store the precision-recall values
    precision = []
    recall = []

    # Loop over the detections
    for i, detection in enumerate(detections):
        bbox = detection[:4]
        confidence = detection[4]

        # Find the best matching ground truth bbox
        max_iou = 0
        match_idx = -1
        for j, gt_bbox in enumerate(ground_truth):
            iou = calculate_iou(bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou
                match_idx = j

        # Assign detection as true positive or false positive
        if max_iou >= iou_threshold:
            if not ground_truth[match_idx][4]:  # If the ground truth is not already matched
                true_positives[i] = 1
                ground_truth[match_idx][4] = 1  # Mark ground truth as matched
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1

        # Calculate precision and recall at the current point
        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = len(ground_truth) - tp
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    # Calculate the average precision using the precision-recall curve
    precision = np.array(precision)
    recall = np.array(recall)
    average_precision = 1/2*(np.sum(precision[:-1] * np.diff(recall)) + np.sum(precision[1:] * np.diff(recall)))
    return average_precision

# # Example usage
# ground_truth = np.array([[10, 10, 100, 100, 0],
#                          [50, 50, 200, 200, 0],
#                          [150, 150, 250, 250, 0]])
# detections = np.array([[10, 10, 100, 100, 0.9],
#                        [120, 120, 200, 200, 0.8],
#                        [50, 50, 200, 200, 0.7],
#                        [
