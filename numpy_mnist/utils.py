# math utils for myCNN

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def max_pool2d(x, size):
    out_height = x.shape[2] // size  # //是整除符号
    out_width = x.shape[3] // size
    pooled = np.zeros((x.shape[0], x.shape[1], out_height, out_width))

    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for i in range(out_height):
                for j in range(out_width):
                    pooled[b, c, i, j] = np.max(
                        x[b, c, i * size:(i + 1) * size, j * size:(j + 1) * size]
                    )

    return pooled


def max_pool2d_backward(dout, x, size):
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


def conv2d(x, weights, bias):
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
                        x[b, :, i:i + kernel_height, j:j + kernel_width] * weights[oc, :, :, :]
                    ) + bias[oc]

    return output


def conv2d_backward(dout, x, weights):
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
                                if i + kh < input_height and j + kw < input_width:
                                    dx[b, ic, i + kh, j + kw] += dout[b, oc, i, j] * weights[oc, ic, kh, kw]
                                dweights[oc, ic, kh, kw] += dout[b, oc, i, j] * x[b, ic, i + kh, j + kw]

    return dx, dweights, dbias


def linear(x, weights, bias):
    return np.dot(x, weights) + bias


def linear_backward(dout, x, weights):
    dx = np.dot(dout, weights.T)
    dweights = np.dot(x.T, dout)
    dbias = np.sum(dout, axis=0)
    return dx, dweights, dbias
