# encoding:utf-8
import torch
from torch.nn import Module
from torch import nn
import gradio as gr
import numpy as np
from PIL import Image
import io

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 1，6，5分别表示输入通道数，输出通道数，卷积核大小
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        # 此处view操作是为了将y的形状从[b, c, h, w]变为[b, c*h*w]，以便送入全连接层
        # -1的涵义是自适应，即自动计算此处应填多少
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


def predict_image(sketch):
    # 预处理图像
    print("entered")
    image = Image.open(io.BytesIO(sketch)).convert("L")
    image = image.convert("L")  # 转换为灰度图
    image = image.resize((28, 28))  # 调整大小为28x28
    image = np.array(image)
    image = image / 255.0  # 归一化
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 调整形状并转换为Tensor
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(image.float())
    return torch.argmax(predictions).item()

def app():
    app = gr.Blocks()

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">MNIST-手写数字识别-Lenet-5</span>
            </div>
            '''
        )

        with gr.Row():
            # 手写数字识别输入鼠标绘制框SketchPad和输出的标签框
            tab_sketchpad_input = gr.Tab(label="SketchPad Input")
            with tab_sketchpad_input:
                input_sketchpad = gr.Sketchpad(label='Draw here')
        with gr.Row():
            tab_label_output = gr.Tab(label="Label Output")
            with tab_label_output:
                output_label = gr.Label(label='Predicted Label')
        input_sketchpad.change(
            fn=predict_image,
            inputs=input_sketchpad,
            outputs=output_label
        )
        app.launch(share=True)


if __name__ == "__main__":
    # 加载模型
    model = torch.load('models/mnist.pth', map_location=torch.device('cpu'))
    model.eval()
    app()

