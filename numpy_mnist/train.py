import mycnn as cnn
import numpy as np

# Load the MNIST dataset

if __name__ == '__main__':
    # 示例数据 (使用随机数据作为示例)
    X = np.random.randn(10, 1, 28, 28)  # 10个样本，1个通道，28x28图像
    y = np.random.randn(10, 10)         # 10个样本，10个类别

    # 创建模型实例
    model = cnn.MyCNN()
    # 训练模型
    model.train(X, y, epochs=10, learning_rate=0.01)
