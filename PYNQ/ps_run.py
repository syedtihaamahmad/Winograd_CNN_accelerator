import numpy as np


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def relu(x):
    return np.maximum(0,x)


def run_fc_computer(x_raw):
    data = '../data/lenet_bias/'

    fc1_w = np.loadtxt(data + "fc1_w.txt", dtype=np.int32, delimiter=',',
                       usecols=[0]).reshape([500, 7*7*20])
    fc1_b = np.loadtxt(data + "fc1_b.txt", dtype=np.int32, delimiter=',',
                       usecols=[0])
    fc2_w = np.loadtxt(data + "fc2_w.txt", dtype=np.int32, delimiter=',',
                       usecols=[0]).reshape([10, 500])
    fc2_b = np.loadtxt(data + "fc2_b.txt", dtype=np.int32, delimiter=',',
                       usecols=[0])

#  x_raw = np.loadtxt(data + "conv2_out.txt", dtype=np.int16,
#                    delimiter=',', usecols=[0],
#                    converters={_: lambda s: int(s, 16) for _ in range(7 * 7 * 10)})
    x_raw = x_raw.reshape([7, 7, 10])

    x = np.zeros(shape=[20, 7, 7], dtype=np.uint8)

    for row in range(7):
        for col in range(7):
            for ch in range(10):
                x[2 * ch, row, col] = np.uint8(x_raw[row, col, ch] & np.uint8(0xff))
                x[2 * ch + 1, row, col] = np.uint8(x_raw[row, col, ch] >> np.uint8(8))

    # x = x.reshape([7, 7, 20])
    # x = np.transpose(x, (1, 2, 0))
    x = x.flatten()

    print(fc1_w.shape, x.shape)
    y = np.matmul(fc1_w, x) + fc1_b
    y = relu(y)
    y = (y / 2**7).round()
    y = y.clip(min=-(2 ** 8 - 1), max=(2 ** 8 - 1))

    y = np.matmul(fc2_w, y) + fc2_b
    y = (y / 2**15)

    # y = log_softmax(y)
#     print("Done...")
    return y






