import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, datatype = "train"):
        self.datatype = datatype
        self.root = '/dtu/datasets1/02514/data_wastedetection/'
        self.anns_file_path = self.root + '/' + 'annotations.json'
        self.coco = COCO(self.anns_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.transforms = transforms.Compose([
            # transforms.PILToTensor(),
            transforms.ToTensor(),
        ])
        
        self.category_id_to_name = {d["id"]: d["name"] for d in self.coco.dataset["categories"]}
        
        # split into train and test
        idxs = np.arange(len(self.ids))
        idxs = np.random.permutation(idxs)
        self.train_idxs = idxs[:int(0.8*len(idxs))]
        self.test_idxs = idxs[int(0.8*len(idxs)):]
        self.train_idxs = self.train_idxs[:int(0.8*len(self.train_idxs))]
        self.val_idxs = self.train_idxs[int(0.8*len(self.train_idxs)):]

        print(f"Number of train images: {len(self.train_idxs)}")
        print(f"Number of val images: {len(self.val_idxs)}")
        print(f"Number of test images: {len(self.test_idxs)}")
        
    def __getitem__(self, idx):
        if self.datatype == "train":
            idx = self.train_idxs[idx]
        elif self.datatype == "val":
            idx = self.val_idxs[idx]
        elif self.datatype == "test":
            idx = self.test_idxs[idx]
            
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[idx]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Labels (In my case, I only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        label_ids, labels = [], []
        for i in range(num_objs):
            labels.append(coco_annotation[i]['category_id'])
            # label_ids.append(coco_annotation[i]['category_id'])
            # labels.append(self.category_id_to_name[coco_annotation[i]['category_id']])
        labels = torch.as_tensor(labels)
        # label_ids = torch.as_tensor(label_ids, dtype=torch.int64)
        
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        # my_annotation["label_ids"] = label_ids
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        # return len(self.ids)
        if self.datatype == "train":
            return len(self.train_idxs)
        elif self.datatype == "val":
            return len(self.val_idxs)
        elif self.datatype == "test":
            return len(self.test_idxs)

def get_dataloader(dataset):
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Batch size
    train_batch_size = 1

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    return data_loader

def show_img(img, annotations, label_dict, ax = None):
    """Show image with annotations"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    ax.imshow(img)
    
    for idx in range(len(annotations["boxes"])):
        box = annotations["boxes"][idx].cpu()
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # add label
        label_id = annotations["labels"][idx].cpu()
        label = label_dict[label_id.item()]
        ax.text(xmin, ymin, f"{label}", fontsize=12, color="r")
        
        
    # plt.show()

if __name__ == "__main__":
    # create own Dataset
    dataset = TacoDataset()
    data_loader = get_dataloader(dataset)
    
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DataLoader is iterable over Dataset
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)
        
        show_img(imgs[0], annotations[0], dataset.category_id_to_name)
        plt.savefig("test.png")
        
        break