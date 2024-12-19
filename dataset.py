# dataset.py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class HandwrittenDigitsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): 包含手写数字数据的CSV文件路径。
            transform (callable, optional): 可选的转换函数。
        """
        # 读取 CSV 文件，不使用列名
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

        # 打印数据形状用于调试
        print(f"数据集大小: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取当前行数据
        row = self.data.iloc[idx].values

        # 分离标签和像素数据
        label = row[0]  # 第一列为标签
        pixels = row[1:]  # 后续列为像素

        # 确认像素长度
        if len(pixels) != 784:
            raise ValueError(f"样本 {idx} 的像素数量为 {len(pixels)}，预期为 784。")

        # 将像素值转换为28x28的图像
        image_np = pixels.reshape(28, 28).astype(np.uint8)

        # 将 numpy 数组转换为 PIL Image
        image = Image.fromarray(image_np, mode='L')  # 'L' 表示灰度图

        if self.transform:
            image = self.transform(image)

        return image, label
