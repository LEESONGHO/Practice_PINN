import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1: CPU

import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
print(device_lib.list_local_devices() )

def set_random_seed(seed):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # Add this line for TensorFlow
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

set_random_seed(7777)

# 0. y_true
def true_solution(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

x_train = np.linspace(0, 0.5, num=11)[:, None].astype(np.float32)
x_train1 = np.linspace(0, 1, num=11)[:, None].astype(np.float32)
y_train_true = true_solution(x_train)
y_train1_true = true_solution(x_train1)

plt.plot(x_train, y_train_true, 'o--')
plt.plot(x_train1, y_train1_true, '^--')


# 1. Governing Equation Loss
def true_solution(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

def compute_derivatives(x, model):
    with tf.GradientTape(persistent=True) as tape4:
        with tf.GradientTape(persistent=True) as tape3:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(x)
                    tape2.watch(x)
                    tape3.watch(x)
                    tape4.watch(x)
                    # true_solution 함수를 호출하여 y 값을 설정
                    # y = true_solution(x)
                    y = model(x)
                    # y = y_pred
                    tape1.watch(y)
                dy_dx = tape1.gradient(y, x)
                tape2.watch(dy_dx)
            d2y_dx2 = tape2.gradient(dy_dx, x)
            tape3.watch(d2y_dx2)
        d3y_dx3 = tape3.gradient(d2y_dx2, x)
        tape4.watch(d3y_dx3)
    d4y_dx4 = tape4.gradient(d3y_dx3, x)

    # del tape1, tape2, tape3, tape4  # 모든 tape를 삭제
    return y, dy_dx, d2y_dx2, d3y_dx3, d4y_dx4

def governing_equation_loss(d4y_dx4):
    f_x = d4y_dx4 + 1
    loss_ge = tf.reduce_mean(tf.square(f_x))
    return loss_ge


# Define a neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,), kernel_initializer='he_normal'),
        tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(1, kernel_initializer='he_normal')
    ])
    return model

# Create the model
model = build_model()
print(model.summary())










