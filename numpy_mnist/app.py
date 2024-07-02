import src.mycnn as cnn
import numpy as np
import gradio as gr
from PIL import Image


def predict_image(sketch):
    # Decode base64 to bytes if necessary
    image = Image.fromarray(sketch).convert("L")
    print("image opened")
    image = image.resize((28, 28))  # 调整大小为28x28
    image = np.array(image)
    image = image / 255.0  # 归一化
    model = cnn.MyCNN()
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    # 使用模型进行预测
    predicted_label = model.forward(image)
    #返回最大值的索引：
    predicted_label = np.argmax(predicted_label).item()
    return predicted_label


def button_click(sketch):
    print("Button clicked")  # 确认按钮被点击
    predicted_label = predict_image(sketch)
    return predicted_label


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
            with gr.Column():
                input_sketchpad = gr.Sketchpad(label='Draw here')
                predict_button = gr.Button("Predict")
                output_label = gr.Label(label='Predicted Label')

        # 按钮点击事件绑定
        predict_button.click(
            fn=button_click,
            inputs=input_sketchpad,
            outputs=output_label
        )
        app.launch(share=True)


if __name__ == '__main__':
    app()

