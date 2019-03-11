import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time
print("Start")
def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.float()
            data = data.to(device)
            target = target.long().to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc


import os

class WSJ():
    """ Load the WSJ speech dataset
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """
  
    def __init__(self):
        self.WSJ = "/home/d.milovanov/win_kaggle/digit"
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(self.WSJ, 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(self.WSJ, 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = np.load(os.path.join(self.WSJ, 'test.npy'), encoding='bytes')
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )


def _with_index_and_stride(x, index_, stride):
    y = np.zeros((x.shape[0] + 2 * stride, x.shape[1]))

    y[stride: y.shape[0] - stride] = x
    y[:stride] = x[::-1][-stride:]
    y[- stride:] = x[::-1][:stride]
    return np.array([np.concatenate(y[_ - stride: _ + stride + 1]) for _ in range(index_, np.shape(y)[0] - stride)])


loader = WSJ()
trainX, trainY = loader.train
valX, valY = loader.dev
testX = loader.test
val_X_pad = np.array([_with_index_and_stride(val, 30,30) for val in valX])
print((val_X_pad).shape)
test_X_pad = np.array([_with_index_and_stride(test, 30,30) for test in testX])
print(test_X_pad.shape)



from torch.utils.data import Dataset, DataLoader
class Dataset_HW1(Dataset):
    
    def __init__(self, x, y):
 
        self.x = np.concatenate(x)
        self.y = np.concatenate(y)
        
    def __len__(self):
        return len(self.x)
      
    def __getitem__(self, idx):

        data = torch.from_numpy(self.x[idx])
        
        if self.y is not None:
            label = torch.from_numpy(np.array(self.y[idx]))
            return data, label
          
        else:
            return data
val_dataset = Dataset_HW1(val_X_pad, valY)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=512, num_workers=0, drop_last=True)

class Simple_MLP(nn.Module):
    def __init__(self):
        super(Simple_MLP, self).__init__()
        self.linear1 = nn.Linear(in_features=2440, out_features=3024)
        self.bn1 = nn.BatchNorm1d(num_features=3024)
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features=3024, out_features=1524)
        self.bn2 = nn.BatchNorm1d(num_features=1524)
        self.drop2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(in_features=1524, out_features=1024)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.linear4 = nn.Linear(in_features=1024, out_features=138)

    def forward(self, input):
        y = F.leaky_relu(self.drop1(self.bn1(self.linear1(input))))
        y = F.leaky_relu(self.drop2(self.bn2(self.linear2(y))))
        y = F.leaky_relu(self.bn3(self.linear3(y)))
        y = F.softmax(self.linear4(y), dim=1)
        return y



model = Simple_MLP()
criterion = nn.CrossEntropyLoss()
print(model)
cuda = torch.cuda.is_available()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")



def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(device)

    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()
        data = data.float()
        data = data.to(device)
        target = target.long().to(device)

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss



n_epochs = 3
Train_loss = []
Test_loss = []
Test_acc = []



for i in range(n_epochs):
    print("==============={0}/{1} epoch".format(i, n_epochs))
    for i in range(0, 14):
        s = np.arange(trainX.shape[0])
        np.random.shuffle(s)
        s = s[:2000]
        trX, trY = trainX[s], trainY[s]
        train_X_pad = np.array([_with_index_and_stride(train, 30,30) for train in trX])
        train_Y_p = trY
        train_dataset = Dataset_HW1(train_X_pad, train_Y_p)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=512, num_workers=0, drop_last=True)
        del train_X_pad
        del train_Y_p
        del trX
        del trY
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test_model(model, val_loader, criterion)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        print('='*20)


from torch.autograd import Variable
model.eval()
count = 0
dict_answ = {}
for utter in test_X_pad:
    utter  = torch.from_numpy(utter.astype(float))
    for sample in range(utter.shape[0]):
        sample_data = Variable(utter[sample:sample+1].clone())
        sample_data = sample_data.type(torch.FloatTensor)
        sample_data = sample_data.cuda()
        sample_out = model(sample_data)
        indices = torch.argmax(sample_out, dim=1)
        dict_answ[count] = indices.item()
        count += 1

import pandas as pd
answ = pd.DataFrame(list(dict_answ.items()), columns=['id', 'label'])
answ.to_csv('submission_{}.csv'.format(12), index=False, header=True)

