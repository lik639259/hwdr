import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_history(history):
    
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12,5))

    # 绘制损失曲线
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], 'g-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'b-', label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], 'g-', label='训练准确率')
    plt.plot(epochs, history['val_acc'], 'b-', label='验证准确率')
    plt.title('训练与验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def check_file_exists(file_path):
    """
    检查指定的文件是否存在。
    
    参数:
    - file_path (str): 文件路径。
    
    返回:
    - bool: 文件是否存在。
    """
    return os.path.exists(file_path)

def create_directory(dir_path):
    """
    创建指定的目录（如果不存在）。
    
    参数:
    - dir_path (str): 目录路径。
    """
    os.makedirs(dir_path, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵。
    
    参数:
    - y_true (list or array): 真实标签。
    - y_pred (list or array): 预测标签。
    - classes (list): 类别名称列表。
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.show()
