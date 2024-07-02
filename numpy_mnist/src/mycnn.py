# basic functions for CNN

import numpy as np
from utils import net as net


class MyCNN:
    def __init__(self):
        # 初始化卷积层参数
        self.cache = {}
        self.conv1_weights = np.random.randn(6, 1, 5, 5) * 0.01
        self.conv1_bias = np.random.randn(6) * 0.01
        self.conv2_weights = np.random.randn(16, 6, 5, 5) * 0.01
        self.conv2_bias = np.random.randn(16) * 0.01

        # 初始化全连接层参数
        self.fc1_weights = np.random.randn(256, 120) * 0.01
        self.fc1_bias = np.random.randn(120) * 0.01
        self.fc2_weights = np.random.randn(120, 84) * 0.01
        self.fc2_bias = np.random.randn(84) * 0.01
        self.fc3_weights = np.random.randn(84, 10) * 0.01
        self.fc3_bias = np.random.randn(10) * 0.01

    def forward(self, x):
        # Conv1
        x = net.conv2d(x, self.conv1_weights, self.conv1_bias)
        self.cache['conv1'] = x
        x = net.relu(x)
        self.cache['relu1'] = x
        x = net.max_pool2d(x, 2)
        self.cache['pool1'] = x

        # Conv2
        x = net.conv2d(x, self.conv2_weights, self.conv2_bias)
        self.cache['conv2'] = x
        x = net.relu(x)
        self.cache['relu2'] = x
        x = net.max_pool2d(x, 2)
        self.cache['pool2'] = x

        # Flatten
        x = x.reshape(x.shape[0], -1)
        self.cache['flatten'] = x

        # FC1
        x = net.linear(x, self.fc1_weights, self.fc1_bias)
        self.cache['fc1'] = x
        x = net.relu(x)
        self.cache['relu3'] = x

        # FC2
        x = net.linear(x, self.fc2_weights, self.fc2_bias)
        self.cache['fc2'] = x
        x = net.relu(x)
        self.cache['relu4'] = x

        # FC3
        x = net.linear(x, self.fc3_weights, self.fc3_bias)
        self.cache['fc3'] = x
        x = net.relu(x)
        self.cache['relu5'] = x

        return x

    def backward(self, dout):
        grads = {}

        # FC3
        dout = net.relu_backward(dout, self.cache['fc3'])
        dout, grads['fc3_weights'], grads['fc3_bias'] = net.linear_backward(dout, self.cache['relu4'],
                                                                            self.fc3_weights)

        # FC2
        dout = net.relu_backward(dout, self.cache['fc2'])
        dout, grads['fc2_weights'], grads['fc2_bias'] = net.linear_backward(dout, self.cache['relu3'],
                                                                            self.fc2_weights)

        # FC1
        dout = net.relu_backward(dout, self.cache['fc1'])
        dout, grads['fc1_weights'], grads['fc1_bias'] = net.linear_backward(dout, self.cache['flatten'],
                                                                            self.fc1_weights)

        # Reshape
        dout = dout.reshape(self.cache['pool2'].shape)

        # Conv2
        dout = net.max_pool2d_backward(dout, self.cache['relu2'], 2)
        dout = net.relu_backward(dout, self.cache['conv2'])
        dout, grads['conv2_weights'], grads['conv2_bias'] = net.conv2d_backward(dout, self.cache['pool1'],
                                                                                self.conv2_weights)

        # Conv1
        dout = net.max_pool2d_backward(dout, self.cache['relu1'], 2)
        dout = net.relu_backward(dout, self.cache['conv1'])
        dout, grads['conv1_weights'], grads['conv1_bias'] = net.conv2d_backward(dout, self.cache['conv1'],
                                                                                self.conv1_weights)

        return grads

    def update_params(self, grads, learning_rate):
        self.conv1_weights -= learning_rate * grads['conv1_weights']
        self.conv1_bias -= learning_rate * grads['conv1_bias']
        self.conv2_weights -= learning_rate * grads['conv2_weights']
        self.conv2_bias -= learning_rate * grads['conv2_bias']
        self.fc1_weights -= learning_rate * grads['fc1_weights']
        self.fc1_bias -= learning_rate * grads['fc1_bias']
        self.fc2_weights -= learning_rate * grads['fc2_weights']
        self.fc2_bias -= learning_rate * grads['fc2_bias']
        self.fc3_weights -= learning_rate * grads['fc3_weights']
        self.fc3_bias -= learning_rate * grads['fc3_bias']

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            # 前向传播
            out = self.forward(x)

            # 计算损失 (假设使用简单的均方误差损失)
            loss = np.mean((out - y) ** 2)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

            # 计算损失的梯度
            dout = 2 * (out - y) / y.size

            # 反向传播
            grads = self.backward(dout)

            # 更新参数
            self.update_params(grads, learning_rate)

