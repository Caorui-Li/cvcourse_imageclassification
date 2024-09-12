import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义模型保存的路径
model_save_path = './model1'
train_results_path ='./result1'
#test_results_path ='./result_test'
#test_model_path = os.path.join(model_save_path, model_save_path)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
# 定义转换器
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),       # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 定义CNN模型
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    print("开始训练")
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        running_loss = 0.0
        for i,(images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 保存每个epoch的模型
        model_path = os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Saved model1 state at epoch {epoch + 1} to {model_path}')            # 保存每个epoch的损失
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        with open(os.path.join(train_results_path, 'train_results.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}\n')
        print(f'Saved train results at epoch {epoch + 1} to {train_results_path}')

# 测试模型
def test_model(model, test_loader,model_path=None):
    print("开始测试")
    model.eval()
    model.to(device)
    if model_path:
        #model.load_state_dict(torch.load(model_path))
        # 加载保存的模型状态
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'model_epoch_10.pth')))
    correct = 0
    total = 0
    test_loader = tqdm(test_loader, desc='Testing', leave=True)
    with torch.no_grad():
        for images, labels in test_loader:
            print(images.shape)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader.set_postfix(accuracy=f'{100 * correct / total:.2f}%')
    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
    with open(os.path.join(train_results_path, 'test_results.txt'), 'a') as f:
        f.write(f'Accuracy of the network on the test images: {100 * correct / total}%')

# 加载数据集
train_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign1\dataset\A", transform=transform)
test_dataset = datasets.ImageFolder(root="E:\second_2\machine vision\\assignment\image_classification\\assign1\dataset\A", transform=transform)
print("train_dataset长度是：")
print(len(train_dataset))
print("test_dataset长度是：")
print(len(test_dataset))
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

model = CatDogCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
# 测试模型
test_model(model, test_loader,model_path='assign1/model1/model_epoch_10.pth')