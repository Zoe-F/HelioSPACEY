# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:19:36 2023

@author: Zoe.Faes
"""

# imports
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import astropy.units as u
import dill as pickle
from datetime import datetime
from random import shuffle
#from timeseries import TimeSeries
torch.cuda.is_available()
#torch.backends.mps.is_available()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
torch.cuda.set_device(device)


file_names = ['Carolina_Almeida_011523_SH_1_20200701_20250701_6h',
              'Daniel_Phoenix_080621_SH_1_20200701_20250701_6h',
              'Daniel_Verscharen_101420_SH_1_20200701_20250701_6h',
              'Hyunjin_Jeong_050422_SH_3_20200701_20250701_6h',
              'Jihyeon_Son_090422_SH_4_20200701_20250701_6h',
              'limei_yan_032322_SH_1_20200701_20250701_6h',
              'limei_yan_032422_SH_1_20200701_20250701_6h',
              'limei_yan_032522_SH_2_20200701_20250701_6h',
              'limei_yan_032522_SH_3_20200701_20250701_6h',
              'Manuel_Grande_062021_SH_1_20200701_20250701_6h',
              'MariaElena_Innocenti_111020_SH_1_20200701_20250701_6h',
              'Michael_Terres_110920_SH_1_20200701_20250701_6h',
              'Michael_Terres_111020_SH_1_20200701_20250701_6h',
              'Ou_Chen_081721_SH_2_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_1_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_2_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_3_20200701_20250701_6h',
              'Qingbao_He_112022_SH_1_20200701_20250701_6h',
              'Qingbao_He_112022_SH_2_20200701_20250701_6h',
              'Sanchita_Pal_041621_SH_1_20200701_20250701_6h',
              'Zoe_Faes_101922_SH_1_20200701_20250701_6h']

file_paths = ['./Timeseries/{}.pickle'.format(name) for name in file_names]

conjs = []
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        # load data
        data = pickle.load(file)
        conjs.append(data[1])

    
start = datetime.now()
    
###################################  DATA  ####################################

# FOR CONJ VS NON-CONJ: set binary=True
# TO CLASSIFY DIFFERENT CATEGORIES OF CONJUNCTIONS: set binary=False

label_labelling = {'non_conj': 0, 'cone': 1, 'quadrature': 2, 'opposition': 3, 'parker spiral': 4}

def get_data(conjunctions, split, binary=True, var='V1', 
             truncate_after=5*u.day, test_method='train/test'):
    
    non_conjs = [c.timeseries[var] for conjs in conjunctions for c in conjs.non_conjs]
    cones = [c.timeseries[var] for conjs in conjunctions for c in conjs.cones]
    quads = [c.timeseries[var] for conjs in conjunctions for c in conjs.quads]
    opps = [c.timeseries[var] for conjs in conjunctions for c in conjs.opps]
    parkers = [c.timeseries[var] for conjs in conjunctions for c in conjs.parkers]

    for s in [non_conjs, cones, quads, opps, parkers]:
        shuffle(s)

    dataset = [[ts for ts in non_conjs if len(ts.data[0]) > 4],
               [ts for ts in cones if len(ts.data[0]) > 4],
               [ts for ts in quads if len(ts.data[0]) > 4],
               [ts for ts in opps if len(ts.data[0]) > 4],
               [ts for ts in parkers if len(ts.data[0]) > 4]]
    stratified_split = [round(split*len(ds)) for ds in dataset]
    
    if test_method == 'train/test':
        train_dataset = [ds[:s] for ds, s in zip(dataset, stratified_split)]
        test_dataset = [ds[s:] for ds, s in zip(dataset, stratified_split)]
        
    #elif test_method == 'cross-validation':
        # TODO: Finish this
    
    tensors = []
    labels = []
    weights = []
    for i, ds in enumerate(train_dataset):
        for ts in ds:
            if binary:
                if i==0:
                    tensor, label = ts.get_truncated_tensor(i, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
                else:
                    tensor, label = ts.get_truncated_tensor(1, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
            else:
                tensor, label = ts.get_truncated_tensor(i, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
            if tensor != None:
                tensors.append(tensor)
                labels.append(label)
        weights.append(len(tensors)) if i==0 else weights.append(len(tensors)-sum(weights[:i-1]))
    for i, tensor in enumerate(tensors):
        if i == 0:
            tens = tensor
        else:
            tens = torch.cat((tens, tensor), 0)
    training_tensors = torch.flatten(tens, 1, -1)
    for i, label in enumerate(labels):
        if i == 0:
            labs = label
        else:
            labs = torch.cat((labs, label), 0)
    training_labels = torch.flatten(labs, 0, -1)
    training_labels = training_labels.type(torch.LongTensor)
    training_labels = training_labels.to(device)
    
    weights = weights/sum(weights)
    
    tensors = []
    labels = []
    for i, ds in enumerate(test_dataset):
        for ts in ds:
            if binary:
                if i==0:
                    tensor, label = ts.get_truncated_tensor(i, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
                else:
                    tensor, label = ts.get_truncated_tensor(1, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
            else:
                tensor, label = ts.get_truncated_tensor(i, new_length=truncate_after, discard_mostly_padded=True, conj=conjunctions[0])
            if tensor != None:
                tensors.append(tensor)
                labels.append(label)
    for i, tensor in enumerate(tensors):
        if i == 0:
            tens = tensor
        else:
            tens = torch.cat((tens, tensor), 0)
    testing_tensors = torch.flatten(tens, 1, -1)
    for i, label in enumerate(labels):
        if i == 0:
            labs = label
        else:
            labs = torch.cat((labs, label), 0)
    testing_labels = torch.flatten(labs, 0, -1)
    testing_labels = testing_labels.type(torch.LongTensor)
    testing_labels = testing_labels.to(device)
    
    print('Training dataset size: ', training_tensors.size())
    print('Test dataset size: ', testing_tensors.size())
    
    return training_tensors, training_labels, testing_tensors, testing_labels, weights
    
    
train_tensors, train_labels, test_tensors, test_labels, weights = get_data(conjs, 0.8, binary=False)

class TimeseriesDataset(Dataset):
    def __init__(self, ts_labels, ts_tensor, transform=None, target_transform=None):
        self.labels = ts_labels
        self.ts_tensor = ts_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ts_tensor)

    def __getitem__(self, idx):
        ts = self.ts_tensor[idx]
        label = self.labels[idx]
        if self.transform:
            ts = self.transform(ts)
        if self.target_transform:
            label = self.target_transform(label)
        
        return ts, label
    
training_data = TimeseriesDataset(train_labels, train_tensors)
test_data = TimeseriesDataset(test_labels, test_tensors)

#################################  START NN  ##################################

learning_rate = 10**(-3)
batch_size = 64
epochs = 100

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(40,30)
        self.layer2 = nn.Linear(30,30)
        self.layer3 = nn.Linear(30,20)
        self.layer4 = nn.Linear(20,20)
        self.layer5 = nn.Linear(20,5)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        output = self.layer5(x)
        return output

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
#loss_fn = nn.NLLLoss(weight=torch.tensor([1., 0.5], device=device))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, TP, FP, FN, TN = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
            for i in range(y.size()[0]):
                if y[i] == logits.argmax(1)[i] and y[i] == 1:
                    TP += 1
                elif y[i] == logits.argmax(1)[i] and y[i] == 0:
                    TN += 1
                elif y[i] != logits.argmax(1)[i] and y[i] == 0:
                    FP += 1
                elif y[i] != logits.argmax(1)[i] and y[i] == 1:
                    FN += 1
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    print('TP: {}, FP: {}, FN: {}, TN: {}.'.format(TP, FP, FN, TN))

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

end = datetime.now()

print(start, end)


# Approx. 30% of conjunctions in each category have a length <= 1 day (4 data points)
# number of conjunctions in each category per sim: 128 217 97 116
# number of conjunctions <= 1 day in each category: 36 75 27 39 ~30%
# number of conjunctions < 3 days in each category: 48 87 40 50 ~40%
# number of conjunctions < 5 days in each category: 52 141 53 73 ~(40%, 65%, 55%, 63%)
# number of conjunctions < 8 days in each category: 64 171 67 91 ~(50%, 80%, 70%, 80%)
# number of conjunctions < 10 days in each category: 74 178 70 95 ~(58%, 82%, 72%, 82%)
