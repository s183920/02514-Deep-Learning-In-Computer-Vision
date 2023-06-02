import torch.nn.functional as F
import torch
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from hotdog.dataloader import HotdogDataset

#We define the training as a function so we can easily re-use it.
def train(model, optimizer, num_epochs=10, loss = "CrossEntropyLoss"):
    trainset = HotdogDataset(train=True)
    train_loader = trainset.get_dataloader(batch_size=32, shuffle=True)
    testset = HotdogDataset(train=False)
    test_loader = testset.get_dataloader(batch_size=32, shuffle=False)
    
    if loss == "CrossEntropyLoss":
        def loss_fun(output, target):
            return F.cross_entropy(output, target)
    else:
        def loss_fun(output, target):
            return F.nll_loss(torch.log(output), target)
        
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            
            if loss.isnan():
                raise ValueError("Loss is NaN")
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict


