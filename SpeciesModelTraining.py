import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
import sklearn.model_selection as skms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as TF
from cv2 import cv2
from sklearn.model_selection import KFold
%matplotlib inline
# define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'gpu'
RANDOM_SEED = 42

#create a set of data for analysis
def picDimensions(img_folder):
    imHeights = []
    imWidths = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir)):
            image = imageio.imread(os.path.join(img_folder, dir, file))
            imHeights.append(image.shape[0])
            imWidths.append(image.shape[1])
    
    return imHeights, imWidths

#find the image with the smallest dimensions
def findSmallest(img_folder):
    smallestHeight = 1000000
    smallestHeightClass = ""
    smallestHeightName = ""
    smallestWidth = 1000000
    smallestWidthClass = "" 
    smallestWidthName = "" 
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir)):
            image = imageio.imread(os.path.join(img_folder, dir, file))
            imHeight = image.shape[0]
            imWidth = image.shape[1]
            if imHeight <= smallestHeight:
                smallestHeight = imHeight
                smallestHeightClass = dir
                smallestHeightName = file
            if imWidth <= smallestWidth:
                smallestWidth = imWidth
                smallestWidthClass = dir
                smallestWidthName = file
    
    print('smallest Height : ' , smallestHeight, smallestHeightClass, smallestHeightName)
    print('smallest Width : ' , smallestWidth, smallestWidthClass, smallestWidthName)
    
    def create_dataset(img_folder):
    img_data_array = []
    class_name = []
    #read and format images
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            image = image.transpose(2,0,1)
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

#displays box plots with distribution of dimensions amoung pictures
Heights, Widths = picDimensions(r'GA_Birds/train')
fig1, ax1 = plt.subplots()
ax1.set_title('Heights')
ax1.boxplot(Heights)
fig2, ax2 = plt.subplots()
ax2.set_title('Widths')
ax2.boxplot(Widths)

#create testing and training sets
train_X, train_y = create_dataset(r'GA_Birds/train')
test_X, test_y = create_dataset(r'GA_Birds/test')
train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)
y = np.append(train_y, test_y)
lookupTable, yNums = np.unique(y, return_inverse= True)
train_yNums = yNums[:train_y.shape[0]]
test_yNums = yNums[train_y.shape[0]:]
print(train_X.shape, train_yNums.shape)


#more formatting and splitting the  data
class DatasetHack:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.y)

shuf = np.arange(train_X.shape[0])
np.random.shuffle(shuf)
split = train_X.shape[0] // 10
split_val = shuf[:split]
split_train = shuf[split:]

train = DatasetHack(train_X[split_train], train_yNums[split_train])
val = DatasetHack(train_X[split_val], train_yNums[split_val])
test = DatasetHack(test_X, test_yNums)
print(type(test[1]))
#validation split

# Hyperparameters
num_epochs = 100
batch_size = 24

# set dataset loader
train_loader = td.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
val_loader = td.DataLoader(dataset=val, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
dataset_sizes = {'train':len(train), 'test':len(test)}


# instantiate the model
model = tv.models.resnet50(num_classes=23).to(DEVICE)
#un-comment these lines if you with to add layers to the end of the model. This tends to only decrease its accuracy
#num_features = model.fc.in_features
#model.fc = nn.Sequential( nn.Dropout(0.5), nn.Linear(num_features, 23)).to(DEVICE)

# instantiate optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

accAtEpochs = list()
# loop over epochs
for epoch in range(num_epochs):
# train the model
    print(epoch)
    model.train()
    train_loss = list()
    train_acc = list()
    for batch in train_loader:
        x, y = batch
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        # predict bird species
        y_pred = model(x)
        # calculate the loss
        loss = F.cross_entropy(y_pred, y)
        # backprop & update weights
        loss.backward()
        optimizer.step()  
        # calculate the accuracy
        acc = accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])
        
        train_loss.append(loss.item())
        train_acc.append(acc)
                 
    # validate the model
    model.eval()
    val_loss = list()
    val_acc = list()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # predict bird species
            y_pred = model(x)
            
            # calculate the loss
            loss = F.cross_entropy(y_pred, y)
            # calculate the accuracy
            acc = accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])
        val_loss.append(loss.item())
        val_acc.append(acc)
    #test model at epochs
    trueAtEpoch = list()
    predAtEpoch = list()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = model(x)
            trueAtEpoch.extend([val.item() for val in y])
            predAtEpoch.extend([val.item() for val in y_pred.argmax(dim=-1)])
    accAtEpochs.append(accuracy_score(trueAtEpoch,predAtEpoch))
    print("acc after epoch " , epoch+1 , ":" , accAtEpochs[epoch])
    #save high performing models
    if accAtEpochs[epoch] >= 0.8:
        torch.save(model.state_dict(), 'checkpoint.pth')
    # adjust the learning rate
    scheduler.step()



# test the model
true = list()
pred = list()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = model(x)
        true.extend([val.item() for val in y])
        pred.extend([val.item() for val in y_pred.argmax(dim=-1)])
# calculate the accuracy 
test_accuracy = accuracy_score(true, pred)
print('Test accuracy: {:.3f}'.format(test_accuracy))
plt.scatter(list(range(1,num_epochs+1)), accAtEpochs)
plt.show()

