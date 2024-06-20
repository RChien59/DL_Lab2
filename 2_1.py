import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


print("Training")
dataset_train_3c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/train/3c', transform = transform)
dataset_train_2c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/train/2c', transform = transform)
dataset_train_1c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/train/1c', transform = transform)

print("val")
dataset_val_3c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/val/3c', transform = transform)
dataset_val_2c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/val/2c', transform = transform)
dataset_val_1c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/val/1c', transform = transform)

print("Testing")
dataset_test_3c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/test/3c', transform = transform)
dataset_test_2c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/test/2c', transform = transform)
dataset_test_1c = ImageFolder(root='/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing/test/1c', transform = transform)



dataloader_train_3c = DataLoader(dataset_train_3c, batch_size=64, shuffle=True)
dataloader_train_2c = DataLoader(dataset_train_2c, batch_size=64, shuffle=True)
dataloader_train_1c = DataLoader(dataset_train_1c, batch_size=64, shuffle=True)

dataloader_val_3c = DataLoader(dataset_val_3c, batch_size=64, shuffle=False)
dataloader_val_2c = DataLoader(dataset_val_2c, batch_size=64, shuffle=False)
dataloader_val_1c = DataLoader(dataset_val_1c, batch_size=64, shuffle=False)

dataloader_test_3c= DataLoader(dataset_test_3c, batch_size=64, shuffle=False)
dataloader_test_2c= DataLoader(dataset_test_2c, batch_size=64, shuffle=False)
dataloader_test_1c= DataLoader(dataset_test_1c, batch_size=64, shuffle=False)

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=10,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class Dynamic_conv2d_first(nn.Module):
    def __init__(self, out_planes, kernel_size, ratio = 0.25, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, K = 10, temperature = 34, init_weight = True):
        super(Dynamic_conv2d_first, self).__init__()

        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention_3c = attention2d(3, ratio, K, temperature)
        self.attention_2c = attention2d(2, ratio, K, temperature)
        self.attention_1c = attention2d(1, ratio, K, temperature)

        self.weight_3c = nn.Parameter(torch.randn(K, out_planes, 3//groups, kernel_size, kernel_size), requires_grad=True)
        self.weight_2c = nn.Parameter(torch.randn(K, 3, 2//groups, kernel_size, kernel_size), requires_grad=True)
        self.weight_1c = nn.Parameter(torch.randn(K, 3, 1//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight_3c[i])
            nn.init.kaiming_uniform_(self.weight_2c[i])
            nn.init.kaiming_uniform_(self.weight_1c[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        batch_size, in_planes, height, width = x.size()
        # print(in_planes)
        
        if in_planes == 1:
            softmax_attention_1c = self.attention_1c(x)
            x1 = x.view(1, -1, height, width)
            weight = self.weight_1c.view(self.K, -1)
            aggregate_weight = torch.mm(softmax_attention_1c, weight).view(batch_size*3, 1//self.groups, self.kernel_size, self.kernel_size)
            x1 = F.conv2d(x1, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            x1 = x1.view(batch_size, 3, x.size(-2), x.size(-1))
            
            softmax_attention_3c = self.attention_3c(x1)
        
        elif in_planes == 2:
            softmax_attention_2c = self.attention_2c(x)
            x2 = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
            weight = self.weight_2c.view(self.K, -1)
            aggregate_weight = torch.mm(softmax_attention_2c, weight).view(batch_size*3, 2//self.groups, self.kernel_size, self.kernel_size)
            x2 = F.conv2d(x2, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            x2 = x2.view(batch_size, 3, x.size(-2), x.size(-1))
            softmax_attention_3c = self.attention_3c(x2)
        else:
            # print(x.shape)
            softmax_attention_3c = self.attention_3c(x)
            
        if in_planes == 1:
            x = x1.repeat(1, 3, 1, 1) + x
        elif in_planes == 2:
            x = x2.repeat(1, 3, 1, 1) + x
        # print(x.shape)
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight_3c.view(self.K, -1)
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention_3c, weight).view(batch_size*self.out_planes, 3//self.groups, self.kernel_size, self.kernel_size)
        # print(aggregate_weight.shape)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention_3c, self.bias).view(-1)
            # print(aggregate_weight.shape)
            # print(x.shape)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = Dynamic_conv2d_first(out_planes=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Dynamic_conv2d(in_planes=16, out_planes=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = Dynamic_conv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Dynamic_conv2d(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 8 * 8 , 1024)
        self.fc2 = nn.Linear(1024, 50)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))
        
        x4 = x4.view(x4.size(0), -1)
        
        x5 = F.relu(self.fc1(x4))
        x6 = self.dropout(x5)
        x7 = self.fc2(x6)
        
        return x7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 檢查是否有 GPU，有則使用 GPU
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

summary(model)
input1 = torch.randn(64, 3, 128, 128).to(device)
flops, params = profile(model, inputs=(input1,))

print(f"FLOPs: {flops}")
print(f"Params: {params}")
breakpoint()

# 訓練模型
num_epochs = 10
train_accuracies = []
for epoch in range(num_epochs):
    model.train()
    total_1c = total_2c = total_3c = 0
    correct_1c = correct_2c = correct_3c = 0
    loss_1c_total = loss_2c_total = loss_3c_total = 0

    dataloader_3c_iter = cycle(dataloader_train_3c)
    
    for i, ((images_1c, labels_1c), (images_2c, labels_2c)) in enumerate(zip(dataloader_train_1c, dataloader_train_2c)):
        
        images_3c, labels_3c = next(dataloader_3c_iter)

        images_1c = images_1c.to(device)
        labels_1c = labels_1c.to(device)

        images_2c = images_2c.to(device)
        labels_2c = labels_2c.to(device)

        images_3c = images_3c.to(device)
        labels_3c = labels_3c.to(device)

         # 訓練 images_1c
        outputs_1c = model(images_1c)
        loss_1c = criterion(outputs_1c, labels_1c)
        optimizer.zero_grad()
        loss_1c.backward()
        optimizer.step()

        _, predicted_1c = outputs_1c.max(1)
        total_1c += labels_1c.size(0)
        correct_1c += predicted_1c.eq(labels_1c).sum().item()
        loss_1c_total += loss_1c.item()

        # 訓練 images_2c
        outputs_2c = model(images_2c)
        loss_2c = criterion(outputs_2c, labels_2c)
        optimizer.zero_grad()
        loss_2c.backward()
        optimizer.step()

        _, predicted_2c = outputs_2c.max(1)
        total_2c += labels_2c.size(0)
        correct_2c += predicted_2c.eq(labels_2c).sum().item()
        loss_2c_total += loss_2c.item()

        # 訓練 images_3c
        outputs_3c = model(images_3c)
        loss_3c = criterion(outputs_3c, labels_3c)
        optimizer.zero_grad()
        loss_3c.backward()
        optimizer.step()

        _, predicted_3c = outputs_3c.max(1)
        total_3c += labels_3c.size(0)
        correct_3c += predicted_3c.eq(labels_3c).sum().item()
        loss_3c_total += loss_3c.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader_train_1c)}], Loss 1c: {loss_1c.item():.4f}, Loss 2c: {loss_2c.item():.4f}, Loss 3c: {loss_3c.item():.4f}')

    # 每個 epoch 結束後輸出總的準確率和平均 loss
    accuracy_1c = 100 * correct_1c / total_1c
    accuracy_2c = 100 * correct_2c / total_2c
    accuracy_3c = 100 * correct_3c / total_3c

    avg_loss_1c = loss_1c_total / (total_1c / images_1c.size(0))
    avg_loss_2c = loss_2c_total / (total_2c / images_2c.size(0))
    avg_loss_3c = loss_3c_total / (total_3c / images_3c.size(0))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy 1c: {accuracy_1c:.2f}%, Loss 1c: {avg_loss_1c:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy 2c: {accuracy_2c:.2f}%, Loss 2c: {avg_loss_2c:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy 3c: {accuracy_3c:.2f}%, Loss 3c: {avg_loss_3c:.4f}')
    '''for i, (images, labels) in enumerate(dataloader_train_1c):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader_train_1c)}], Loss: {loss.item():.4f}')
    
    for i, (images, labels) in enumerate(dataloader_train_2c):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader_train_2c)}], Loss: {loss.item():.4f}')

    for i, (images, labels) in enumerate(dataloader_train_3c):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader_train_3c)}], Loss: {loss.item():.4f}')
'''


    # 驗證模型
    model.eval()
    val_loss_1c = 0
    correct_1c = 0
    total_1c = 0
    val_loss_2c = 0
    correct_2c = 0
    total_2c = 0
    val_loss_3c = 0
    correct_3c = 0
    total_3c = 0
    with torch.no_grad():
        for images, labels in dataloader_val_1c:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_1c += loss.item()
            _, predicted = outputs.max(1)
            total_1c += labels.size(0)
            correct_1c += predicted.eq(labels).sum().item()

        for images, labels in dataloader_val_2c:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_2c += loss.item()
            _, predicted = outputs.max(1)
            total_2c += labels.size(0)
            correct_2c += predicted.eq(labels).sum().item()

        for images, labels in dataloader_val_3c:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_3c += loss.item()
            _, predicted = outputs.max(1)
            total_3c += labels.size(0)
            correct_3c += predicted.eq(labels).sum().item()

    print(f'Validation Loss: {val_loss_1c / len(dataloader_val_1c):.4f}, Accuracy: {100. * correct_1c / total_1c:.2f}%')
    print(f'Validation Loss: {val_loss_2c / len(dataloader_val_1c):.4f}, Accuracy: {100. * correct_2c / total_2c:.2f}%')
    print(f'Validation Loss: {val_loss_3c / len(dataloader_val_1c):.4f}, Accuracy: {100. * correct_3c / total_3c:.2f}%')

# 繪製訓練accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()
plt.show()

model.eval()
test_loss_1c = 0
correct_1c = 0
total_1c = 0
test_loss_2c = 0
correct_2c = 0
total_2c = 0
test_loss_3c = 0
correct_3c = 0
total_3c = 0
with torch.no_grad():
    for images, labels in dataloader_test_1c:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss_1c += loss.item()
        _, predicted = outputs.max(1)
        total_1c += labels.size(0)
        correct_1c += predicted.eq(labels).sum().item()

    for images, labels in dataloader_test_2c:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss_2c += loss.item()
        _, predicted = outputs.max(1)
        total_2c += labels.size(0)
        correct_2c += predicted.eq(labels).sum().item()

    for images, labels in dataloader_test_3c:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss_3c += loss.item()
        _, predicted = outputs.max(1)
        total_3c += labels.size(0)
        correct_3c += predicted.eq(labels).sum().item()

print(f'Test Loss: {test_loss_1c / len(dataloader_test_1c):.4f}, Accuracy: {100. * correct_1c / total_1c:.2f}%')
print(f'Test Loss: {test_loss_2c / len(dataloader_test_2c):.4f}, Accuracy: {100. * correct_2c / total_2c:.2f}%')
print(f'Test Loss: {test_loss_3c / len(dataloader_test_3c):.4f}, Accuracy: {100. * correct_3c / total_3c:.2f}%')
