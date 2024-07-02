import tensorflow as tf
import sys
sys.path.insert(1, '../gradio')
import gradio
from tensorflow.keras.layers import *


(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1,784) / 255.0, x_test.reshape(-1,784) / 255.0


def get_trained_model(n):
    model = tf.keras.models.Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(784,)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train[:n], y_train[:n], epochs=2)
    print(model.evaluate(x_test, y_test))
    return model


model = get_trained_model(n=50000)

# Gradio code #
sketchpad = gradio.Sketchpad(flatten=True, sample_inputs=x_test[:10])
label = gradio.outputs.Label()
io = gradio.Interface(inputs=sketchpad, outputs=label, model=model, model_type="keras", verbose=False,
                      always_flag=True)
httpd, path_to_local_server, share_url = io.launch(inline=True, share=True, inbrowser=True)

print("URL for MNIST model interface: ", share_url)