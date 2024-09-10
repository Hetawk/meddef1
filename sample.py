import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch import nn

from loader.dataset_loader import DatasetLoader

# 数据预处理和加载
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪和缩放到224x224大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色调整
    transforms.RandomRotation(degrees=10),  # 随机旋转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


# transform_test = transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop((224,224)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

train_dataset = torchvision.datasets.ImageFolder(root='./dataset/scisic/Train',transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root='./dataset/scisic/Test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

datasets_dict = DatasetLoader.get_all_datasets()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # Determine device based on GPU availability


model = models.resnet50(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the model on the test images: {100 * correct / total}%")
