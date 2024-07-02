import numpy as np

class Model:
    def __init__(self):
        # 初始化卷积层参数
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

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_backward(self, dout, x):
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx
    
    def max_pool2d(self, x, size):
        out_height = x.shape[2] // size
        out_width = x.shape[3] // size
        pooled = np.zeros((x.shape[0], x.shape[1], out_height, out_width))
        
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for i in range(out_height):
                    for j in range(out_width):
                        pooled[b, c, i, j] = np.max(
                            x[b, c, i*size:(i+1)*size, j*size:(j+1)*size]
                        )
        
        return pooled

    def max_pool2d_backward(self, dout, x, size):
        dx = np.zeros_like(x)
        
        out_height = x.shape[2] // size
        out_width = x.shape[3] // size
        
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for i in range(out_height):
                    for j in range(out_width):
                        (h_start, h_end) = (i * size, (i + 1) * size)
                        (w_start, w_end) = (j * size, (j + 1) * size)
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        max_value = np.max(pool_region)
                        dx[b, c, h_start:h_end, w_start:w_end] = (pool_region == max_value) * dout[b, c, i, j]
        
        return dx

    def conv2d(self, x, weights, bias):
        out_channels, in_channels, kernel_height, kernel_width = weights.shape
        batch_size, _, input_height, input_width = x.shape
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = np.zeros((batch_size, out_channels, output_height, output_width))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        output[b, oc, i, j] = np.sum(
                            x[b, :, i:i+kernel_height, j:j+kernel_width] * weights[oc, :, :, :]
                        ) + bias[oc]
        
        return output

    def conv2d_backward(self, dout, x, weights):
        out_channels, in_channels, kernel_height, kernel_width = weights.shape
        batch_size, _, input_height, input_width = x.shape
        _, _, output_height, output_width = dout.shape
        
        dx = np.zeros_like(x)
        dweights = np.zeros_like(weights)
        dbias = np.zeros(out_channels)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        dbias[oc] += dout[b, oc, i, j]
                        for ic in range(in_channels):
                            for kh in range(kernel_height):
                                for kw in range(kernel_width):
                                    dx[b, ic, i+kh, j+kw] += dout[b, oc, i, j] * weights[oc, ic, kh, kw]
                                    dweights[oc, ic, kh, kw] += dout[b, oc, i, j] * x[b, ic, i+kh, j+kw]
        
        return dx, dweights, dbias

    def linear(self, x, weights, bias):
        return np.dot(x, weights) + bias
    
    def linear_backward(self, dout, x, weights):
        dx = np.dot(dout, weights.T)
        dweights = np.dot(x.T, dout)
        dbias = np.sum(dout, axis=0)
        return dx, dweights, dbias

    def forward(self, x):
        self.cache = {}
        
        # Conv1
        x = self.conv2d(x, self.conv1_weights, self.conv1_bias)
        self.cache['conv1'] = x
        x = self.relu(x)
        self.cache['relu1'] = x
        x = self.max_pool2d(x, 2)
        self.cache['pool1'] = x
        
        # Conv2
        x = self.conv2d(x, self.conv2_weights, self.conv2_bias)
        self.cache['conv2'] = x
        x = self.relu(x)
        self.cache['relu2'] = x
        x = self.max_pool2d(x, 2)
        self.cache['pool2'] = x
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        self.cache['flatten'] = x
        
        # FC1
        x = self.linear(x, self.fc1_weights, self.fc1_bias)
        self.cache['fc1'] = x
        x = self.relu(x)
        self.cache['relu3'] = x
        
        # FC2
        x = self.linear(x, self.fc2_weights, self.fc2_bias)
        self.cache['fc2'] = x
        x = self.relu(x)
        self.cache['relu4'] = x
        
        # FC3
        x = self.linear(x, self.fc3_weights, self.fc3_bias)
        self.cache['fc3'] = x
        x = self.relu(x)
        self.cache['relu5'] = x
        
        return x
    
    def backward(self, dout):
        grads = {}
        
        # FC3
        dout = self.relu_backward(dout, self.cache['fc3'])
        dout, grads['fc3_weights'], grads['fc3_bias'] = self.linear_backward(dout, self.cache['relu4'], self.fc3_weights)
        
        # FC2
        dout = self.relu_backward(dout, self.cache['fc2'])
        dout, grads['fc2_weights'], grads['fc2_bias'] = self.linear_backward(dout, self.cache['relu3'], self.fc2_weights)
        
        # FC1
        dout = self.relu_backward(dout, self.cache['fc1'])
        dout, grads['fc1_weights'], grads['fc1_bias'] = self.linear_backward(dout, self.cache['flatten'], self.fc1_weights)
        
        # Reshape
        dout = dout.reshape(self.cache['pool2'].shape)
        
        # Conv2
        dout = self.max_pool2d_backward(dout, self.cache['relu2'], 2)
        dout = self.relu_backward(dout, self.cache['conv2'])
        dout, grads['conv2_weights'], grads['conv2_bias'] = self.conv2d_backward(dout, self.cache['pool1'], self.conv2_weights)
        
        # Conv1
        dout = self.max_pool2d_backward(dout, self.cache['relu1'], 2)
        dout = self.relu_backward(dout, self.cache['conv1'])
        dout, grads['conv1_weights'], grads['conv1_bias'] = self.conv2d_backward(dout, self.cache['conv1'], self.conv1_weights)
        
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

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # 前向传播
            out = self.forward(X)
            
            # 计算损失 (假设使用简单的均方误差损失)
            loss = np.mean((out - y) ** 2)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')
            
            # 计算损失的梯度
            dout = 2 * (out - y) / y.size
            
            # 反向传播
            grads = self.backward(dout)
            
            # 更新参数
            self.update_params(grads, learning_rate)
            
# 示例数据 (使用随机数据作为示例)
X = np.random.randn(10, 1, 28, 28)  # 10个样本，1个通道，28x28图像
y = np.random.randn(10, 10)         # 10个样本，10个类别

# 创建模型实例
model = Model()

# 训练模型
model.train(X, y, epochs=10, learning_rate=0.01)