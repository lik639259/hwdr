import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import ImprovedResNet18  # 修改为新的模型
from dataset import HandwrittenDigitsDataset
from util import check_file_exists, plot_confusion_matrix

def load_model(model_path='models/best_model.pth'):
    if not os.path.exists(model_path):
        print(f'找不到模型文件 {model_path}，请先训练模型。')
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedResNet18().to(device)  # 使用新的模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(model)  # 打印模型结构
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
        transforms.Normalize((0.1307,), (0.3081,))  # 确保与训练时一致
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批量维度
    return image

def visualize_preprocessed_image(image_tensor):
    image_np = image_tensor.squeeze().cpu().numpy()
    plt.imshow(image_np, cmap='gray')
    plt.title('预处理后的图像')
    plt.show()

def predict(model, device, image):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        print(f'模型输出 (未归一化): {outputs}')
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        print(f'预测概率分布: {probabilities}')
        _, predicted = torch.max(outputs.data, 1)
    visualize_preprocessed_image(image)
    return predicted.item()

def main():
    if len(sys.argv) < 2:
        print("用法: python predict.py <image_path1> <image_path2> ...")
        sys.exit(1)
    
    model, device = load_model()
    image_paths = sys.argv[1:]
    for image_path in image_paths:
        image = preprocess_image(image_path)
        if image is None:
            continue
        prediction = predict(model, device, image)
        print(f'图像: {image_path} -> 预测数字: {prediction}')

if __name__ == "__main__":
    main()