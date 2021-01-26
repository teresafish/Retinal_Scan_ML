import math
import torch
import ijson
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH = 1
EPOCHS = 4

class DRData(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.resize = transforms.Resize((512, 512))

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    img = Image.open(img_path + '.jpg')
    DR_label = torch.tensor(int(self.annotations.iloc[index, 1]))
    ME_label = torch.tensor(int(self.annotations.iloc[index, 2]))
    
    img = self.resize(img)

    if self.transform:
      img = self.transform(img)
    
    return img, DR_label, ME_label

class network(nn.Module):
  def __init__(self):
    super(network, self).__init__()

    # convolution and pooling
    self.conv1a = nn.Conv2d(3, 64, 3, padding = 1)
    self.conv1b = nn.Conv2d(64, 64, 3, padding = 1)

    self.conv2a = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv2b = nn.Conv2d(128, 128, 3, padding = 1)

    self.conv3a = nn.Conv2d(128, 256, 3, padding = 1)
    self.conv3b = nn.Conv2d(256, 256, 3, padding = 1)

    self.conv4 = nn.Conv2d(256, 512, 3, padding = 1)
    self.conv5 = nn.Conv2d(512, 512, 3, padding = 1)

    self.pool = nn.MaxPool2d(2, 2)

    self.flatten = nn.Flatten(0, 0)

    self.fc1 = nn.Linear(1048576, 2048)
    self.fc2 = nn.Linear(2048, 2048)
    self.fc3a = nn.Linear(2048, 10) # for DR grading
    self.fc3b = nn.Linear(2048, 4)  # for ME grading

    self.sig = nn.Sigmoid()

  def forward(self, x):

    x = self.conv1a(x)
    x = self.conv1b(x)
    x = self.pool(F.relu(x))

    x = self.conv2a(x)
    x = self.conv2b(x)
    x = self.pool(F.relu(x))

    x = self.conv3a(x)
    x = self.conv3b(x)
    x = self.conv3b(x)
    x = self.pool(F.relu(x))

    #x = self.conv4(x)
    #x = self.conv5(x)
    #x = self.conv5(x)
    #x = self.conv5(x)
    #x = self.pool(F.relu(x))

    #for i in range(4):
    #  x = self.conv5(x)

    x = torch.flatten(x, 1, 3)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    DR = self.fc3a(x)
    ME = self.fc3b(x)

    return DR, ME


class lossFunc(nn.Module):
  def __init__(self):
    super(lossFunc, self).__init__()
    self.DR_criterion = nn.CrossEntropyLoss()
    self.ME_criterion = nn.CrossEntropyLoss()

  def forward(self, DR_predict, ME_predict, DR_target, ME_target):
    DRLoss = self.DR_criterion(DR_predict, DR_target)
    MELoss = self.ME_criterion(ME_predict, ME_target)
    return DRLoss + MELoss



if __name__ == '__main__':
  train_set = DRData(csv_file = 'train_labels.csv', root_dir = 'Training', transform = transforms.ToTensor())
  print('train set done')
  test_set = DRData(csv_file = 'valid_labels.csv', root_dir = 'Validation', transform = transforms.ToTensor())
  print('test set done')

  train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = BATCH)
  print('train load done')
  test_loader = DataLoader(dataset = train_set, shuffle = False, batch_size = BATCH)
  print('train load done')

  # Getting NN model and optimiser
  net = network()
  optimiser = optim.Adam(net.parameters(), lr = 0.001)
  loss = lossFunc()
  # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  DR_stages = ('None','Minimal','Mild-Moderate','Severe','Proliferative')
  CSME_present = ('None', 'Mild', 'Moderate', 'Severe')

  for epoch in range(EPOCHS):
    running_loss = 0
    for i, data in enumerate(train_loader):
      image, DR_target, ME_target = data
      print('got data')
      # zero the parameter gradients
      optimiser.zero_grad()

      # forward + backward + optimize
      DR_predict, ME_predict = net(image)
      print('got prediction')
      loss_total = loss(DR_predict, ME_predict, DR_target, ME_target)
      loss_total.backward()
      optimizer.step()

      print(i)

      # print statistics
      running_loss += loss_total.item()

      if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
  
  print('Finished Training')

  PATH = './DR_net.pth'
  torch.save(net.state_dict(), PATH)

  net.load_state_dict(torch.load(PATH))

  correctDR_only = 0
  correctME_only = 0
  correctBoth = 0

  with torch.no_grad():
    for batch in test_loader:
      inputs = pixel_field[batch.pixel_data]
      DR_target = batch.DR_grade
      ME_target = batch.ME_grade

      DR_out, ME_out = convertNetOutput(net(inputs))

      correctDR = DR_grade == DR_out.flatten()
      correctME = ME_grade == ME_out.flatten()

      correctDR_only += torch.sum(correctDR & ~correctME).item()
      correctME_only += torch.sum(correctME & ~correctDR).item()
      correctBoth += torch.sum(correctDR & correctME).item()

  correctDROnlyPercent = correctDR_only / len(test_set)
  correctMEOnlyPercent = correctME_only / len(test_set)
  bothCorrectPercent = correctBoth / len(test_set)
  neitherCorrectPer = 1 - correctDROnlyPercent - correctMEOnlyPercent - bothCorrectPercent

  print(neitherCorrectPer)

  score = 100 * (bothCorrectPercent 
                 + 0.5 * correctDROnlyPercent
                 + 0.5 * correctMEOnlyPercent)
 
  print("\n"
              "ME incorrect, DR incorrect: {:.2%}\n"
              "ME correct, DR incorrect: {:.2%}\n"
              "ME incorrect, DR correct: {:.2%}\n"
              "ME correct, DR correct: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}".format(neitherCorrectPer,
                                              correctMEOnlyPercent,
                                              correctDROnlyPercent,
                                              bothCorrectPercent, score))

