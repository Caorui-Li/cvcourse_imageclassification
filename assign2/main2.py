import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义模型保存的路径
model_save_path = './model2'
train_results_path ='./result2'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图像大小以匹配ResNet输入
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),       # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        # 加载预训练的ResNet模型
        self.resnet = models.resnet18(pretrained=True)
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print("开始训练")
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        running_loss = 0.0
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 保存每个epoch的模型
        model_path = os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Saved model2 state at epoch {epoch + 1} to {model_path}')

        # 保存每个epoch的损失
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        with open(os.path.join(train_results_path, 'train_results.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}, Loss: {running_loss/len(train_loader)}\n')
        print(f'Saved train results at epoch {epoch + 1} to {train_results_path}')


def test_model(model, test_loader, device,model_path=None):
    print("开始测试")
    model.eval()
    model.to(device)
    if model_path:
        # 加载保存的模型状态
        #model.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'model_epoch_10.pth')))
    correct = 0
    total = 0
    test_loader = tqdm(test_loader, desc='Testing', leave=True)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
    with open(os.path.join(train_results_path, 'test_results.txt'), 'a') as f:
        f.write(f'Accuracy of the network on the test images: {100 * correct / total}%')


train_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign2\A", transform=transform)
test_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign2\A", transform=transform)


# 加载数据集
train_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign1\dataset\A", transform=transform)
test_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign1\dataset\A", transform=transform)
print("train_dataset长度是：")
print(len(train_dataset))
print("test_dataset长度是：")
print(len(test_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

model =CustomResNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer,device, num_epochs=10)
test_model(model, test_loader, device, model_path='assign2/model2/model_epoch_10.pth')

