# train.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

from model import ImprovedResNet18
from dataset import HandwrittenDigitsDataset
from util import plot_training_history, check_file_exists, create_directory

def train_model(csv_path, model_save_path='models/best_model.pth',
               num_epochs=50, batch_size=128, learning_rate=0.001, patience=5):
    # 数据预处理，加入数据增强
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),        # 随机旋转±10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 确保与模型预处理一致
    ])

    # 加载训练数据集
    dataset = HandwrittenDigitsDataset(csv_file=csv_path, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% 用于训练
    val_size = total_size - train_size   # 20% 用于验证

    # 打印拆分信息用于调试
    print(f"总数据集大小: {total_size}")
    print(f"训练集大小 (80%): {train_size}")
    print(f"验证集大小 (20%): {val_size}")

    # 数据集拆分
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedResNet18().to(device)
    print(model)  # 打印模型结构

    # 定义损失函数和优化器，加入L2正则化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 引入学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1, patience=3, verbose=True)

    # 训练过程中的记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / train_size
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_size
        epoch_val_acc = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # 学习率调度器更新
        scheduler.step(epoch_val_loss)

        # 早停检测
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            print('验证损失改善，模型已保存。')
        else:
            epochs_no_improve += 1
            print(f'验证损失未改善，当前耐心值: {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print('早停触发。训练结束。')
                break

    # 绘制训练历史
    plot_training_history(history)

    print('训练完成，最佳模型已保存。')

def main():
    train_csv_path = 'train_data.csv'
    if not check_file_exists(train_csv_path):
        print(f'找不到 {train_csv_path} 文件，请确保文件存在于当前目录下。')
        return
    # 确保models目录存在
    create_directory('models')
    train_model(train_csv_path)

if __name__ == "__main__":
    main()