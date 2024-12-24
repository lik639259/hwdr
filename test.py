# test.py
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import ImprovedResNet18  # 导入新的模型类
from dataset import HandwrittenDigitsDataset
from util import plot_confusion_matrix, check_file_exists, create_directory

def load_model(model_path='models/best_model.pth'):
    if not os.path.exists(model_path):
        print(f'找不到模型文件 {model_path}，请先训练模型。')
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedResNet18().to(device)  # 实例化新的模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(model)  # 打印模型结构以确认
    print("模型已加载并设置为评估模式。")
    return model, device

def preprocess_image(image_path):
    if not check_file_exists(image_path):
        print(f'找不到图像文件 {image_path}。')
        return None
    
    try:
        image = Image.open(image_path).convert('L')  # 转为灰度图
    except Exception as e:
        print(f'打开图像文件 {image_path} 时出错: {e}')
        return None
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 与训练时一致
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批量维度
    return image

def visualize_preprocessed_image(image_tensor):
    image_np = image_tensor.squeeze().cpu().numpy()
    plt.imshow(image_np, cmap='gray')
    plt.title('预处理后的图像')
    plt.show()

def test_model(model, device, test_loader):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f'测试集损失: {avg_loss:.4f}, 测试集准确率: {accuracy:.4f}')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, classes=range(10))

def main():
    if len(sys.argv) < 2:
        print("用法: python test.py <test_csv_path>")
        sys.exit(1)
    
    test_csv_path = sys.argv[1]
    if not check_file_exists(test_csv_path):
        print(f'找不到测试CSV文件 {test_csv_path}。')
        sys.exit(1)
    
    model, device = load_model()
    
    # 数据预处理，测试时通常不需要数据增强
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = HandwrittenDigitsDataset(csv_file=test_csv_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    test_model(model, device, test_loader)

if __name__ == "__main__":
    main()
