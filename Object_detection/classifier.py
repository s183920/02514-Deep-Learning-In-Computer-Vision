try:
    from Object_detection.hparams import default_classifier_config, wandb_defaults
    from Object_detection.selective_search import selective_search, rect_coordinates
    from Object_detection.data import TacoDataset, get_dataloader
    from Object_detection.utils import download_model
    from Object_detection.model import models
except ModuleNotFoundError:
    from hparams import default_classifier_config, wandb_defaults
    from selective_search import selective_search, rect_coordinates
    from data import TacoDataset, get_dataloader
    from utils import download_model
    from model import models

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.utils.data
from datetime import datetime as dt
import os
import wandb
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

background_class = 0

class TacoClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, datatype = "train", img_size = (128, 128), ss_size = None, k1 = 0.5, k2 = 0.5):
        super().__init__()

        self.ss_size = ss_size
        self.datatype = datatype
        self.k1 = k1
        self.k2 = k2
        self.img_size = img_size

        self.length = None
        self.org_data = TacoDataset(datatype = self.datatype, img_size = None, length = self.length)
        self.category_id_to_name = self.org_data.category_id_to_name
        self.category_id_to_name[background_class] = "background"

        size_str = str(self.ss_size[0]) if self.ss_size is not None else "none"
        self.data_path = "Object_detection/cls_data/" + datatype + "_" + size_str + ".pkl"
        if os.path.exists(self.data_path):
            print("Loading proposal boxes")
            self.load_boxes()
        else:
            print("Creating proposal boxes")
            self.create_boxes()

        self.create_data()


        # self.transforms = transforms.Compose([
        #     transforms.Resize((128, 128)),
        #     # transforms.ToTensor(),
        # ])

    def create_data(self):
        print("Creating images and labels")


        self.labels = []
        self.data = []

        for idx, data in tqdm(enumerate(self.org_data), total = len(self.org_data)):
            img, gt_ann = data
            img_org = self.org_data.get_img(gt_ann["image_id"].item())
            img_org = self.org_data.transforms(img_org)

            # get selective search boxes
            proposal_boxes = self.proposal_boxes[gt_ann["image_id"].item()]

            # get ious
            ious = torchvision.ops.box_iou(proposal_boxes, gt_ann["boxes"])

            # select foreground and background boxes
            max_ious, box_idxs = torch.max(ious, dim = 1)
            foreground_boxes = proposal_boxes[max_ious >= self.k1]
            background_boxes = proposal_boxes[max_ious < self.k2]

            if len(foreground_boxes) > 0:
                for i, box in enumerate(foreground_boxes):
                    xmin, ymin, xmax, ymax = box
                    xmax = min(xmax+1, img_org.shape[1])
                    ymax = min(ymax+1, img_org.shape[2])
                    # if ((xmax-xmin) == 0) or ((ymax-ymin) == 0):
                    #     continue
                    proposal_img = img_org[:, xmin:xmax, ymin:ymax]
                    if 0 in proposal_img.shape:
                        continue
                    self.data.append(proposal_img)
                    self.labels.append(gt_ann["labels"][box_idxs[i]])
                    # assert xmax-xmin != 0 and ymax-ymin != 0, "Box is 0"

                for box_idx in torch.randperm(len(background_boxes))[:len(foreground_boxes)*3]:
                    xmin, ymin, xmax, ymax = background_boxes[box_idx]
                    xmax = min(xmax+1, img_org.shape[1])
                    ymax = min(ymax+1, img_org.shape[2])
                    # if ((xmax-xmin) == 0) or ((ymax-ymin) == 0):
                    #     continue
                    proposal_img = img_org[:, xmin:xmax, ymin:ymax]
                    if 0 in proposal_img.shape:
                        continue
                    self.data.append(proposal_img)
                    # l = torch.max(list(gt_ann["labels"].values())) +1
                    self.labels.append(torch.tensor(background_class))
                    # assert xmax-xmin != 0 and ymax-ymin != 0, "Box is 0"
        
    def create_boxes(self):
        self.proposal_boxes = {}

        org_data = TacoDataset(datatype = self.datatype, img_size = self.ss_size, length = self.length)

        for idx, data in tqdm(enumerate(org_data), total = len(org_data)):
            img, gt_ann = data
            # img_org = self.org_data.get_img(gt_ann["image_id"].item())
            # img_org = self.org_data.transforms(img_org)

            proposal_boxes = selective_search(img)
            proposal_boxes[:, 2] += proposal_boxes[:, 0]
            proposal_boxes[:, 3] += proposal_boxes[:, 1]

            print(gt_ann["width_scale"], gt_ann["height_scale"])

            proposal_boxes[:, 0] = (proposal_boxes[:, 0] / gt_ann["width_scale"]).astype(int)
            proposal_boxes[:, 1] = (proposal_boxes[:, 1] / gt_ann["height_scale"]).astype(int)
            proposal_boxes[:, 2] = (proposal_boxes[:, 2] / gt_ann["width_scale"]).astype(int)
            proposal_boxes[:, 3] = (proposal_boxes[:, 3] / gt_ann["height_scale"]).astype(int)

            proposal_boxes = torch.tensor(proposal_boxes, dtype = torch.int32)

            self.proposal_boxes[gt_ann["image_id"].item()] = proposal_boxes

        # save data
        os.makedirs(os.path.dirname(self.data_path), exist_ok = True)
        with open(self.data_path, "wb") as f:
            pickle.dump(self.proposal_boxes, f)


    def load_boxes(self):
        with open(self.data_path, "rb") as f:
            self.proposal_boxes = pickle.load(f)

        print("Loaded data from", self.data_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # return self.transforms(self.data[idx]), self.labels[idx]
        # print(idx)
        # print(self.data[idx].shape)
        # print(self.labels[idx])
        return transforms.functional.resize(self.data[idx], self.img_size, antialias=True) , self.labels[idx]

class TacoClassifier:
    def __init__(self, project = "Taco", name = None, model = None, config = None, use_wandb = True, verbose = True, show_test_images = False, model_save_freq = None, **kwargs):
        """
        Class for training and testing a taco classifier

        Parameters
        ----------
        name : str, optional
            Name of the run, by default 'run_' + current time
        config : dict, optional
            Dictionary with configuration, by default the defualt_config from hparams.py
        use_wandb : bool, optional
            Whether to use wandb, by default True
        verbose : bool, optional
            Whether to print progress, by default True
        show_test_images : bool, optional
            Whether to show test images, by default False
        model_save_freq : int, optional
            How often to save the model, by default None
            None means only save the best model
        **kwargs : dict
            Additional arguments to config
            
        Config
        ------
        num_epochs : int
            Number of epochs to train
        dropout : float
            Dropout rate
        batchnorm : bool
            Whether to use batchnormalisation
        train_dataset_kwargs : dict
            Additional arguments to TacoDataset for training
            i.e. train_dataset_kwargs = {"data_augmentation": False}
        test_dataset_kwargs : dict
            Additional arguments to TacoDataset for testing
        optimizer : str
            Optimizer to use (from torch.optim)
        optimizer_kwargs : dict
            Additional arguments to optimizer
            e.g. optimizer_kwargs = {"lr": 0.01}
        scheduler : bool
            Whether to use a scheduler - will use ExponentialLR with gamma = 0.1
            Decrease will happen after 20 % of epochs
        """

        # set info
        self.name = name if name is not None else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_test_images = show_test_images
        self.model_save_freq = model_save_freq

        # set init values
        self.config = config if config is not None else default_classifier_config
        self.config.update(kwargs)
        self.dev_mode = False
        self.wandb_run = None
        self.test_images = []

        # set model
        # print(f"Setting model to {model}")
        self.set_model("Resnet50" if model is None else model)


        # set wandb
        if use_wandb:
            self.set_logger(project = project)


        # set dataset
        self.set_dataset()
        
    
    def load_model(self, path, model_name = "model.pth"):
        print(f"Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = "logs/Taco/models/latest_from_wandb"
            path = path[6:]
            print(f"Downloading model from wandb: {path}")
            path = download_model(path, dl_path)
        
        self.model.load_state_dict(torch.load(path+"/"+model_name, map_location=torch.device(self.device)))           
        
    def set_model(self, model):
        print(f"Setting model to {model}")
        
        model = models.get(model.lower())
        if model is None:
            raise ValueError(f"Model not found")
      
        self.model = model(finetune = self.config["finetune"])

        self.model.to(self.device)


    def set_dataset(self):
        params = {"ss_size": self.config["ss_size"], "k1" : self.config["k1"], "k2" : self.config["k2"], "img_size":self.config["classification_size"]}
        self.data_train = TacoClassifierDataset(datatype = "train", **params)
        self.data_test = TacoClassifierDataset(datatype = "test", **params)
        self.data_val = TacoClassifierDataset(datatype = "val", **params)

        self.train_loader = torch.utils.data.DataLoader(self.data_train,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=4)
        
        self.test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=32, shuffle=False, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(self.data_val, batch_size=32, shuffle=False, num_workers=4)
  
    def set_optimizer(self):
        optimizer = self.config.get("optimizer")
        
        # set default lr if not set
        if optimizer.lower() == "sgd" and "lr" not in self.config:
            self.config["lr"] = 0.01
        elif optimizer.lower() == "adam" and "lr" not in self.config:
            self.config["lr"] = 0.001
            
        # set optimizer
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), lr = self.config["lr"])
   

    def save_model(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(self.name + "_model", type="model", description=f"model trained on taco dataset after {epoch} epochs")
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)

    def set_logger(self, **kwargs):
        # overwrite defaults with parsed arguments
        wandb_settings = wandb_defaults.copy()
        wandb_settings.update(kwargs)
        wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + "/" + self.name}) # create log dir
        wandb_settings.update({"name" : self.name, "group": self.model.name}) # set run name

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb
        self.wandb_run = wandb.init(**wandb_settings, config = self.config)

        # setup wandb config
        self.config = self.wandb_run.config

        # watch model
        self.wandb_run.watch(self.model)


    def prepare(self, cuda_device):
        # set seed
        if self.config.get("seed") is not None:
            torch.manual_seed(self.config.get("seed", 0))

        # set visible cuda devices
        if self.device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip("[]").replace(" ", "")

        # set loss function
        self.loss_fun = lambda output, target: F.cross_entropy(output, target)
        # loss_fun = self.config.get("loss_fun")
        # if loss_fun == "CrossEntropyLoss":
        #     self.loss_fun = lambda output, target: F.cross_entropy(output, target)
        # else:
        #     self.loss_fun = lambda output, target: F.nll_loss(torch.log(output), target)

        # set optimizer
        self.set_optimizer()
        
        # set best acc
        self.best_val_acc = 0


    def train_step(self, data, target):
        # send data to device
        data, target = data.to(self.device), target.to(self.device)

        #Zero the gradients computed for each weight
        self.optimizer.zero_grad()

        #Forward pass your image through the network
        output = self.model(data).float() # kat her

        #Compute the loss
        loss = self.loss_fun(output, target)

        #Backward pass through the network
        loss.backward()

        #Update the weights
        self.optimizer.step()

        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct = (target==predicted).sum().cpu().item()

        return train_correct, loss


    def train(self, num_epochs=None, cuda_device = [0]):
        # prepare training
        num_epochs = self.config.get("num_epochs") if num_epochs is None else num_epochs
        self.prepare(cuda_device)
        print(f"Starting training on {self.device.type}")

        for epoch in tqdm(range(num_epochs), unit='epoch'):
            train_correct = 0
            train_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for minibatch_no, (data, target) in pbar:
                train_correct_, train_loss_ = self.train_step(data, target)
                train_correct += train_correct_
                train_loss += train_loss_.item()

                # break if dev mode
                if self.dev_mode:
                    break

            #Comput the train accuracy
            train_loss /= len(self.train_loader)
            train_acc = train_correct/len(self.data_train)*100
            if self.verbose:
                print("Accuracy train: {train:.1f}%".format(train=train_acc))
            
            
            # test 
            val_acc, val_loss = self.test(validation=True)            
            
            
            # Save model
            if self.model_save_freq is None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_model(f"logs/Taco/models/{self.name}/model.pth", epoch)
            elif epoch % self.model_save_freq == 0:
                self.save_model(f"logs/Taco/models/{self.name}/model.pth", epoch)
            
            
            # log to wandb
            if self.wandb_run is not None:
                # self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log({
                    "Train metrics/train_acc":    train_acc,
                    "Train metrics/train_loss":   train_loss,
                    "Validation metrics/val_acc":     val_acc,
                    "Validation metrics/val_loss":    val_loss,
                    "epoch":        epoch,
                })
            
            # clear cache
            self.clear_cache()
            
        # log best val acc
        if self.wandb_run is not None:
            self.wandb_run.log({"Best validation accuracy": self.best_val_acc}, commit = True)
            
            
    def test(self, validation = False):               
        if validation:
            data_loader = self.val_loader
        else:
            self.prepare([0])
            data_loader = self.test_loader
        data_len = len(data_loader.dataset)
        
        print("Performing test with {} images".format(data_len))
        
        # Init counters
        test_correct = 0
        test_loss = 0
        
        
        # Test the model
        self.model.eval()
        for data, target in data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            predicted = output.argmax(1).cpu()

            # Update counters
            test_correct += (target==predicted).sum().item()
            test_loss += self.loss_fun(output.cpu(), target).item()
    
        # compute stats
        test_acc = test_correct/data_len*100
        test_loss /= len(data_loader)

        
        
        # calculate confusion matrix      
        test_acc = test_correct/data_len*100
        test_loss /= len(data_loader)

        
        if self.verbose:
            print("Accuracy test: {test:.1f}%".format(test=test_acc))


        return test_acc, test_loss  


    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")
        



if __name__ == "__main__":
    classifier = TacoClassifier(project="TacoClassifier", name = "Resnet50", 
                                show_test_images=False, model = "resnet", use_wandb=True, 
                                num_epochs=100,
                                optimizer = "Adam",)

    classifier.train()

    # classifier.test(save_images=10)

    # dataset = TacoClassifierDataset(ss_size=(100, 100), datatype="train")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # for i, (data, target) in enumerate(dataloader):
    #     print(i)
    #     print(data.shape)
    #     print(target)

    # for i in range(len(dataset)):
    #     print(i)
    #     print(dataset[i][0].shape)
    #     print(dataset[i][1])