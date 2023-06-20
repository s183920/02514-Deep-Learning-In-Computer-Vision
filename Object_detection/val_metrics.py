import torch

def IoU(prediction, ground_truth):
    intersection = torch.max(torch.min(prediction[:, 0], ground_truth[:, 0]) * torch.min(prediction[:, 1], ground_truth[:, 1]), torch.zeros(ground_truth.shape[0]))
    union = prediction[:, 0] * prediction[:, 1] + ground_truth[:, 0] * ground_truth[:, 1] - intersection
    return intersection / union

def Average_Precision(confidence_scores, bounding_boxes, ground_truth):
    '''
    Average Precission
    
    input: 
        confidence_scores: confidence scores of prediction
        bounding_boxes: bounding boxes of prediction
        ground_truth: ground truth

    output:
        AP: Average Precission
    '''
    # get prediction
    prediction = torch.cat((confidence_scores.unsqueeze(1), bounding_boxes), dim=1)
    # sort prediction by confidence score
    prediction = prediction[prediction[:, 0].argsort(descending=True)]
    # get ground truth
    ground_truth = ground_truth[ground_truth[:, 0].argsort(descending=True)]
    # get number of ground truth
    num_ground_truth = ground_truth.shape[0]
    # get number of prediction
    num_prediction = prediction.shape[0]
    # get number of classes

    # get true positive
    true_positive = torch.zeros(num_prediction)
    for i in range(num_prediction):
        # get prediction
        prediction_i = prediction[i]
        # get ground truth
        ground_truth_i = ground_truth[ground_truth[:, 1] == prediction_i[1]]
        # get iou
        iou = IoU(prediction_i[2:], ground_truth_i[:, 2:])
        # get max iou
        max_iou, max_iou_index = torch.max(iou, dim=0)
        # get max iou index
        max_iou_index = max_iou_index[max_iou > 0.5]
        # if max iou index is not empty
        if max_iou_index.shape[0] > 0:
            # get max iou index
            max_iou_index = max_iou_index[0]
            # if max iou index is not used
            if ground_truth_i[max_iou_index, 0] == 0:
                # set true positive to 1
                true_positive[i] = 1
                # set ground truth to used
                ground_truth_i[max_iou_index, 0] = 1

    # get cumulative sum of true positive
    true_positive_cumsum = torch.cumsum(true_positive, dim=0)
    # get cumulative sum of prediction
    prediction_cumsum = torch.cumsum(torch.ones(num_prediction), dim=0)
    # get precision
    precision = true_positive_cumsum / prediction_cumsum
    # get recall
    recall = true_positive_cumsum / num_ground_truth
    # get average precission
    AP = torch.sum(precision * true_positive) / num_ground_truth

    return AP, precision, recall
