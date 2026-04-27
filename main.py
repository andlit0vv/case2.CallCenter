import random
import numpy as np
import struct

with open("mnistrain", "rb") as f:
    magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
    image_data = np.fromfile(f, dtype=np.uint8, count=num_images * rows * cols)
    images = image_data.reshape(num_images, rows * cols)

with open("mnilabel", "rb") as f:
    magic, label = struct.unpack('>II', f.read(8))
    labels = np.fromfile(f, dtype=np.uint8)

with open("10k_test_images", "rb") as f:
    magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
    image_data = np.fromfile(f, dtype=np.uint8, count=num_images * rows * cols)
    images_test = image_data.reshape(num_images, rows * cols)

with open("10k_test_labels", "rb") as f:
    magic, label = struct.unpack('>II', f.read(8))
    test_labels = np.fromfile(f, dtype=np.uint8)

test_pib = np.zeros((10000, 10))
test_pib[np.arange(10000), test_labels] = 1
test_labels = test_pib
images_test = images_test.astype(float)/255
pib = np.zeros((60000, 10))
pib[np.arange(60000), labels] = 1

labels = pib
images = images.astype(np.float32)/255.0
E = 2.718281828045
ETA = 0.01
EPOCHS = 20000
LOSS_THRESHOLD = 0.01

def ReLU(x):
    return np.maximum(0, x)

def softmax_b(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, target):
    safe_predict = np.clip(predict, 1e-15, 1.0)
    return -np.sum(target * np.log(safe_predict))

def cross_entropy_loss_b(predict, target):
    safe_predict = np.clip(predict, 1e-15, 1.0)
    per_example_loss = -np.sum(target * np.log(safe_predict), axis=1)
    return np.mean(per_example_loss)

common_loss = []
INPUT_DIM = 784
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 1024
OUTPUT_DIM = 10


W_h1 = np.random.uniform(-0.0576, 0.0576, size=(INPUT_DIM, HIDDEN_DIM1))
B_h1 = np.random.uniform(-0.1, 0.1, size=HIDDEN_DIM1)
W_h2 = np.random.uniform(-0.054, 0.054, size=(HIDDEN_DIM1, HIDDEN_DIM2))
B_h2 = np.random.uniform(-0.1, 0.1, size=HIDDEN_DIM2)
W_o = np.random.uniform(-0.076, 0.076, size=(HIDDEN_DIM2, OUTPUT_DIM))
B_o = np.random.uniform(-0.1, 0.1, size=OUTPUT_DIM)
counter = 0

for epoch in range(EPOCHS):
    k = random.randint(0, 59970)
    x = images[k:k+30]
    y_true = labels[k:k+30]
    logits_1 = (x @ W_h1) + B_h1
    hiden1 = ReLU(logits_1)
    logits_2 = (hiden1 @ W_h2) + B_h2
    hiden2 = ReLU(logits_2)
    logits_3 = (hiden2 @ W_o) + B_o
    predict = softmax_b(logits_3)
    loss = cross_entropy_loss_b(predict, y_true)
    nabla_o = ETA * (hiden2.T @ (predict - y_true))
    W_o -= nabla_o/30
    nabla_bo = ETA * (predict - y_true)
    B_o -= np.mean(nabla_bo, axis=0)
    nabla_h2 = ETA * (hiden1.T  @ (((predict - y_true) @ W_o.T) * (logits_2 > 0).astype(float)))
    W_h2 -= nabla_h2/30
    nabla_bh2 = ETA * (((predict - y_true) @ W_o.T) * (logits_2 > 0).astype(float))
    B_h2 -= np.mean(nabla_bh2, axis=0)
    nabla_h1 = ETA * x.T @ ((((predict - y_true) @ W_o.T) @ W_h2.T) * (logits_2 > 0).astype(float) * (logits_1 > 0).astype(float))
    W_h1 -= nabla_h1/30
    nabla_bh1 = ETA * ((((predict - y_true) @ W_o.T) @ W_h2.T) * (logits_2 > 0).astype(float) * (logits_1 > 0).astype(float))
    B_h1 -= np.mean(nabla_bh1, axis=0)
c_print = 0
for (x, y) in zip(images_test, test_labels):
    c_print += 1
    logits_1 = (x @ W_h1) + B_h1
    hiden1 = ReLU(logits_1)
    logits_2 = (hiden1 @ W_h2) + B_h2
    hiden2 = ReLU(logits_2)
    logits_3 = (hiden2 @ W_o) + B_o
    predict = softmax(logits_3)
    predict = [round(float(p), 5) for p in predict]
    if list(y).index(1) != predict.index(max(predict)):
        counter += 1
print(counter)
np.save("W_h1.npy", W_h1)
np.save("W_h2.npy", W_h2)
np.save("B_o.npy", B_o)
np.save("W_o.npy", W_o)
np.save("B_h1.npy", B_h1)
np.save("B_h2.npy", B_h2)
