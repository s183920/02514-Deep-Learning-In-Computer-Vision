from classifier import TacoClassifier, TacoClassifierDataset
from data import TacoDataset, get_dataloader
from selective_search import selective_search, draw_rectangles, non_maximum_suppression
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
import cv2
from torchvision import transforms
from copy import copy
import numpy as np
import torchvision
import torch
import pickle
from val_metrics import calculate_average_precision
import os

def resize_img(img, img_size = (500, 500)):
    img = transforms.ToPILImage()(img)
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height 
    target_width = int(img_size[0]) 
    target_height = int(img_size[0] / aspect_ratio) if img_size is not None else img_height
    if target_height > img_size[1]:
        target_height = int(img_size[1])
        target_width = int(img_size[1] * aspect_ratio)

    print(f"Resizing image from {img.size} to {(target_width, target_height)}")
    img = img.resize((target_width, target_height), resample=Image.LANCZOS) # Image.ANTIALIAS)

    width_scale = target_width / img_width if img_size is not None else 1.0
    height_scale = target_height / img_height if img_size is not None else 1.0

    img = transforms.ToTensor()(img)
    return img, (width_scale, height_scale)

# def resize_boxes(boxes, scale):
#     boxes[:, 0] = (boxes[:, 0] / scale[0]).astype(int)
#     boxes[:, 1] = (boxes[:, 1] / scale[1]).astype(int)
#     boxes[:, 2] = (boxes[:, 2] / scale[0]).astype(int)
#     boxes[:, 3] = (boxes[:, 3] / scale[1]).astype(int)
#     return boxes

# def draw_boxes(image, boxes, labels = None):
#     # Create a copy of the original image
#     image = image.cpu().numpy().transpose(1, 2, 0)
#     image_copy = image.copy()
#     # image_copy = image.copy()
#     # image_copy = transforms.ToPILImage()(image)
#     if image_copy.dtype == "float32":
#         image_copy *= 255

#     # Draw rectangles on the image
#     for idx, box in enumerate(boxes):
#         x1, y1, x2, y2 = box
#         # x, y, w, h = int(x), int(y), int(w), int(h)
#         cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if labels is not None:
#             cv2.putText(image_copy, labels[idx], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Convert the image from BGR to RGB
#     # image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    
#     return image_copy.astype("uint8")

class ObjectDetector:
    def __init__(self) -> None:
        # model = "wandb:deepcomputer/TacoClassifier/Resnet50_model:v19"
        # model = "wandb:deepcomputer/TacoClassifier/Resnet50_model:v36"
        self.name = "ObjectDetector"
        model = "wandb:deepcomputer/TacoClassifier/Resnet50_model:latest"
        self.classifier = TacoClassifier(use_wandb=False, data = None, model="resnet")
        self.classifier.load_model(model)
        self.result_path = "Object_detection/results/"
        os.makedirs(self.result_path, exist_ok=True)

        with open("category_id_to_name.pkl", "rb") as f:
            self.category_id_to_name = pickle.load(f)

    def get_results(self, img, boxes):
        # get box scores
        self.classifier.model.eval()
        unfilter_boxes = copy(boxes)

        confidence = []
        boxes = []
        labels = []
        background_confidence = []
        background_boxes = []
        background_labels = []
        box_preds = []
        background_box_preds = []
        for box in unfilter_boxes:
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            prop_img = img[:, box[0]:box[2], box[1]:box[3]].unsqueeze(0).to(self.classifier.device)
            if 0 in prop_img.shape:
                continue
            
            pred = torch.exp(self.classifier.model(prop_img))
            if pred.argmax().item() == 0:
                background_confidence.append(pred.max().item())
                background_boxes.append(list(box))
                background_labels.append(self.category_id_to_name[pred.argmax().item()] + f" {pred.max().item():.2f}")
                background_box_preds.append(pred)
            else:
                confidence.append(pred.max().item())
                boxes.append(list(box))
                labels.append(self.category_id_to_name[pred.argmax().item()] + f" {pred.max().item():.2f}")
                box_preds.append(pred)

        confidence = torch.tensor(confidence)
        boxes = torch.tensor(boxes)
        background_confidence = torch.tensor(background_confidence)
        background_boxes = torch.tensor(background_boxes)
        box_preds = torch.stack(box_preds).squeeze(1)
        background_box_preds = torch.stack(background_box_preds).squeeze(1)

        return box_preds, confidence, boxes, labels, background_box_preds, background_confidence, background_boxes, background_labels

    def plot_classified_boxes(self, img, boxes, labels, label_indices, plot_idx):
        fig, ax = plt.subplots()
        # ax.imshow(draw_boxes(img, boxes_nms, labels = [labels[i] for i in nms_indices]))
        ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
        labs = [labels[i] for i in label_indices]
        color = "red"
        for box, label in zip(boxes, labs):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)
            ax.text(x1, y1, label, bbox=dict(facecolor='red', alpha=0.5))
            # ax.text(x1, y1, label, bbox=dict(facecolor='red', alpha=0.5))

        ax.set_axis_off()
        ax.set(title="Classified Proposals (NMS)")

        plt.savefig(self.result_path + self.name + "_" + plot_idx + "_boxes_classified.png")
        plt.show()

    def plot_proposals(self, img, boxes, boxes_nms, plot_idx):
        # plot proposals
        fig, axes = plt.subplots(1, 3)

        ax = axes.flatten()[0]
        ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
        ax.set_axis_off()
        ax.set(title="Image")

        ax = axes.flatten()[1]
        # ax.imshow(draw_boxes(img, boxes))
        ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
        color = "red"
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)
        ax.set_axis_off()
        ax.set(title="Proposals (SS)")

        ax = axes.flatten()[2]
        # ax.imshow(draw_boxes(img, boxes_nms))
        ax.imshow(img.cpu().numpy().transpose(1, 2, 0))
        for box in boxes_nms:
            x1, y1, x2, y2 = box
            rect = Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)
        ax.set_axis_off()
        ax.set(title="Proposals (NMS)")

        plt.savefig(self.result_path + self.name + "_" + plot_idx + "_boxes.png")
        plt.show()

    def make_plots(self, img, boxes, boxes_nms, nms_indices, labels, plot_idx):
        boxes = boxes.cpu().numpy()
        boxes_nms = boxes_nms.cpu().numpy()

        # plot proposals
        self.plot_proposals(img, boxes, boxes_nms, plot_idx)

        # plot classified boxes
        self.plot_classified_boxes(img, boxes_nms, labels, nms_indices, plot_idx)

        

        

    def get_scores(self, img, boxes, box_preds, gt_ann = None):
        box_preds = box_preds.detach().cpu()
        AP = calculate_average_precision(gt_ann["boxes"], boxes, box_preds.argmax(dim = 1), iou_threshold=0.2)
        
        aps = []
        for i in range(1, 29):
            ap = calculate_average_precision(gt_ann["boxes"], boxes, box_preds[:, i])
            aps.append(ap)
        mAP = np.mean(aps)

        return AP, mAP

    def detect(self, img, gt_ann = None, plot = True, plot_name = "object_detector"):
        img, scale = resize_img(img)

        # get proposals
        boxes = selective_search(img, max_proposals=100)
        boxes = torch.tensor(boxes)

        # get box classifier scores
        box_preds, confidence, boxes, labels, background_box_preds, background_confidence, background_boxes, background_labels = self.get_results(img, boxes)

        # filter boxes
        boxes_nms, nms_indices = non_maximum_suppression(boxes.numpy(), confidence.numpy(), 0.25)
        boxes_nms = torch.tensor(boxes_nms)
        print("Reduced from {} to {} boxes".format(len(boxes), len(boxes_nms)))

        if len(boxes) == 0:
            raise Exception("No boxes found")

        # plot boxes
        if plot:
            self.make_plots(img, boxes, boxes_nms, nms_indices, labels, plot_name)

        # get scores
        if gt_ann is not None:
            ap, mAP = self.get_scores(img, boxes, box_preds, gt_ann)        
            return ap, mAP

if __name__ == "__main__":
    from time import time
    from tqdm import tqdm

    od = ObjectDetector()
    dataset = TacoDataset(datatype="test", img_size=None)
    data_loader = get_dataloader(dataset)

    APs = []
    mAPs = []

    i = 0
    
    # pbar = tqdm(total=len(data_loader))
    start = time()
    # for imgs, annotations in data_loader:
    for imgs, annotations in tqdm(data_loader):
        for img, annotation in zip(imgs, annotations):
            if i < 10:
                plot = True
            else:
                plot = False

            try:
                AP, mAP = od.detect(img, annotation, plot = plot, plot_name = "object_detector_{}".format(i))
            except:
                continue
            APs.append(AP)
            mAPs.append(mAP)
            i += 1
        #     break
        # break
    end = time()
    print("Time: {:.2f} s".format(end - start))

    print("AP: {:.2f}".format(np.mean(APs)))
    print("mAP: {:.2f}".format(np.mean(mAPs)))