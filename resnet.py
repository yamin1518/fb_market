import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
import pathlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 16
num_classes = 6
learning_rate = 0.0001
momentum = 0.9


transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=batch_size, shuffle=True
)


root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

model = models.resnet50(pretrained = True)
model.fc = nn.Linear(2048, num_classes)



for param in model.fc.parameters():
    param.requires_grad = False
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum =momentum)



num_epochs = 25
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(1, num_epochs + 1):

    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (inputs_, labels_) in enumerate(train_loader):
        inputs_, labels_ = inputs_.to(device), labels_.to(device)
        optimizer.zero_grad()

        outputs = model(inputs_)
        loss = criterion(outputs, labels_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == labels_).item()
        total += labels_.size(0)
        if (batch_idx) % 500 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                  .format(epoch, num_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.2f}, train-acc: {(100 * correct / total):.2f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        model.eval()
        y_true = np.array([])
        y_pred = np.array([])
        for inputs_t, targets_t in (test_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = model(inputs_t)

            loss_t = criterion(outputs_t, targets_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == targets_t).item()
            y_true = np.concatenate((y_true, targets_t.cpu().numpy()))
            y_pred = np.concatenate((y_pred, pred_t.cpu().numpy()))

            total_t += targets_t.size(0)

        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.3f}, validation acc: {(100 * correct_t / total_t):.3f}\n')


    model.train()


y_pred = y_pred.tolist()
y_true = y_true.tolist()
cm = confusion_matrix(y_true, y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(cm, annot=True, fmt=".1f")
plt.title('Confusion Matrix')
ax.set_xticklabels(['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise'])
ax.set_yticklabels(['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise'])
plt.xticks(rotation=45)
plt.yticks(rotation = 45)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_true, y_pred, target_names=classes))

plt.figure(figsize=(10, 7))
plt.title('Accuracy')
plt.plot(train_acc, color='red', label='Train Accuracy')
plt.plot(val_acc, color='blue', label='Validataion Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()


plt.figure(figsize=(10, 7))
plt.title('Loss')
plt.plot(train_loss, color='red', label='Train loss')
plt.plot(val_loss, color='blue', label='Validataion loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()