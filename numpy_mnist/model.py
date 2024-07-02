from numpy_mnist.src import mycnn as cnn
import numpy as np
import gradio as gr
from PIL import Image
import io


def predict_image(sketch):
    # 预处理图像
    print("entered")
    try:
        image = Image.open(io.BytesIO(sketch)).convert("L")
        image = image.convert("L")  # 转换为灰度图
        image = image.resize((28, 28))  # 调整大小为28x28
        image = np.array(image)
        image = image / 255.0  # 归一化
        model = cnn.MyCNN()
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        # 使用模型进行预测
        predicted_label = model.forward(image)
        return predicted_label
    except Exception as e:
        print(f"Error in predict_image: {e}")  # 输出错误信息
        return None


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

