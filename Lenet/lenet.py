import torch
import torchvision
import torchvision.datasets
import torch.nn as nn
import torchvision.transforms as transforms

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet,self).__init__()
    self.features=nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2),
        nn.Conv2d(6,16,5),
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2),
    )
    self.classifier=nn.Sequential(
        nn.Linear(400,120),
        nn.Tanh(),
        nn.Linear(120,84),
        nn.Tanh(),
        nn.Linear(84,10)
    )
  def forward(self,x):
    #print(x.shape)
    x=self.features(x)
    #print(x.shape)
    x=x.view(x.size(0),-1)
    #print(x.shape)
    x=self.classifier(x)
    #print(x.shape)
    return x

batch_size=128

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transforms.ToTensor())

trainLoader=torch.utils.data.DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.ToTensor())

testloader=torch.utils.data.DataLoader(dataset=testset,batch_size=batch_size,shuffle=False)

Lenet=LeNet()

def eval(dataloader):
  total,correct=0,0
  for data in dataloader:
    inputs,label=data
    outputs=Lenet(inputs)
    _,pred=torch.max(outputs.data,1)
    total+=label.size(0)
    correct+=(pred==label).sum().item()
    return 100*correct/total

eval(trainLoader)

import torch.optim as optim
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(Lenet.parameters())

loss_arr=[]
loss_epoch_arr=[]
epochs=20
for epoch in range(epochs):
  for i,data in enumerate(trainLoader):
    inputs,label=data
    
    outputs=Lenet(inputs)
    loss=loss_fn(outputs,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_arr.append(loss.item())
  print("Loss : | "+str(loss.item())+" | epoch : |"+str(epoch)+" | batch : |" \
        +str(i)+"| train accuracy : |"+str(eval(trainLoader))+"| test accuracy : |"+str(eval(testloader))+"|")
  loss_epoch_arr.append(loss.item())

"""Training The Network on GPU"""

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(dataloader):
  total,correct=0,0
  for data in dataloader:
    inputs,label=data
    inputs,label=inputs.to(device),label.to(device)
    outputs=Lenet(inputs)
    _,pred=torch.max(outputs.data,1)
    total+=label.size(0)
    correct+=(pred==label).sum().item()
    return 100*correct/total

Lenet=LeNet().to(device)

loss_arr=[]
loss_epoch_arr=[]
epochs=20
for epoch in range(epochs):
  for i,data in enumerate(trainLoader):
    inputs,label=data
    inputs,label=inputs.to(device),label.to(device)
    
    outputs=Lenet(inputs)
    loss=loss_fn(outputs,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_arr.append(loss.item())
  print("Loss : | "+str(loss.item())+" | epoch : |"+str(epoch)+" | batch : |" \
        +str(i)+"| train accuracy : |"+str(eval(trainLoader))+"| test accuracy : |"+str(eval(testloader))+"|")
  loss_epoch_arr.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()

