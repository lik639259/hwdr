import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms
from model import ImprovedResNet18
import numpy as np
import os
from tkinter import messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("手写数字识别")
        self.resizable(0, 0)
        
        # 设置画布大小
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)
        
        # PIL 图片用于保存绘制内容
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # 按钮：识别
        self.predict_button = Button(self, text="识别", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=2, padx=2)
        
        # 按钮：清除
        self.clear_button = Button(self, text="清除", command=self.clear)
        self.clear_button.grid(row=1, column=1, pady=2, padx=2)
        
        # 标签：显示结果
        self.result_label = Label(self, text="预测结果: ")
        self.result_label.grid(row=1, column=2, pady=2, padx=2)
        
    def paint(self, event):
        # 绘制线条
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
    
    def clear(self):
        # 清除画布
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='white')
        self.result_label.config(text="预测结果: ")
    
    def preprocess_image(self):
        # 将绘制的图像转换为模型输入所需的格式
        # 缩放到28x28
        img = self.image.resize((28, 28), Image.ANTIALIAS)
        # 反转颜色
        img = ImageOps.invert(img)
        # 转换为灰度
        img = img.convert('L')
        
        # 定义与训练时相同的预处理
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 增加批量维度
        return img_tensor

    def load_model(self):
        # 加载训练好的模型
        model = ImprovedResNet18()
        model_path = 'models/best_model.pth'
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"找不到模型文件 {model_path}，请先训练模型。")
            self.quit()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    
    def predict(self):
        # 进行预测并显示结果
        model, device = self.load_model()
        img_tensor = self.preprocess_image().to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_digit = predicted.item()
            self.result_label.config(text=f"预测结果: {predicted_digit}")

if __name__ == "__main__":
    app = App()
    app.mainloop() 