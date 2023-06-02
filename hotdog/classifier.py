import torch
from torch.nn import functional as F
from model import models
from dataloader import HotdogDataset
import tqdm
import os

class HotdogClassifier:
    def __init__(self, model = None, optimizer = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_model(model)
        self.set_optimizer(optimizer)
        self.set_dataset()
        
    def set_model(self, model):
        model = models.get(model) if model is not None else models.get("SimpleCNN")
        self.model = model()
        self.model.to(self.device)
        
        
    def set_dataset(self):
        self.data_train = HotdogDataset()
        self.data_test = HotdogDataset(train=False)    
        self.train_loader = self.data_train.get_dataloader()
        self.test_loader = self.data_test.get_dataloader()
    
    def set_optimizer(self, optimizer, *args, **kwargs):
        # self.optimizer = optimizer(self.model.parameters())
        optimizer = optimizer if optimizer is not None else "Adam"
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), *args, **kwargs)
        
        
    def train(self, num_epochs=10, loss = "CrossEntropyLoss", cuda_device = [0]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip("[]").replace(" ", "")
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            #For each epoch
            train_correct = 0
            self.model.train()
            for minibatch_no, (data, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                #Zero the gradients computed for each weight
                self.optimizer.zero_grad()
                #Forward pass your image through the network
                output = self.model(data)
                #Compute the loss
                loss = F.nll_loss(torch.log(output), target)
                #Backward pass through the network
                loss.backward()
                #Update the weights
                self.optimizer.step()
                
                #Compute how many were correctly classified
                predicted = output.argmax(1)
                train_correct += (target==predicted).sum().cpu().item()
                
                
            #Comput the train accuracy
            train_acc = train_correct/len(self.data_train)
            print("Accuracy train: {train:.1f}%".format(train=100*train_acc))
            
            # test 
            self.test()
            
            
            
    def test(self):
        #Comput the test accuracy
        test_correct = 0
        self.model.eval()
        for data, target in self.test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target==predicted).sum().item()
        
        test_acc = test_correct/len(self.data_test)
        # print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
        print("Accuracy test: {test:.1f}%".format(test=100*test_acc))
        
        
if __name__ == "__main__":
    classifier = HotdogClassifier()
    classifier.train(num_epochs=5)