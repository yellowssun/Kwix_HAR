import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


## data를 가져오는 코드
data_loc = 'https://github.com/dknife/ML/raw/main/data/'
df = pd.read_csv(data_loc+'nonlinear.csv')
X = df['x'].to_numpy()
y_label = df['y'].to_numpy()

## 모델 구조를 함수화 하였다. 
## Learning rate와 activation을 바꿔줘야 한다.
def layer_model(activation, learning_rate):
    model = keras.models.Sequential( [
    keras.layers.Dense(4, activation=activation),
    keras.layers.Dense(4, activation= activation),
    keras.layers.Dense(4, activation= activation),
    keras.layers.Dense(1, activation= activation),
    ])

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model


## 모데를 학습하고 결과를 출력하는 함수
def plot_result(model, X, y_label, df):
    model.fit(tf.expand_dims(X, axis=-1), y_label, epochs=100)
    domain = np.linspace(0, 1, 100).reshape(-1,1) 
    y_hat = model.predict(domain)
    plt.scatter(df['x'], df['y'])
    plt.scatter(domain, y_hat, color='r')
    plt.show()


"""기존 모델 구조"""
## Loss: 0.3633
model = keras.models.Sequential( [
    keras.layers.Dense(6, activation= 'sigmoid'),
    keras.layers.Dense(4, activation= 'sigmoid'),
    keras.layers.Dense(1, activation= 'sigmoid'),
])

optimizer = keras.optimizers.SGD(learning_rate=5.0)
model.compile(optimizer=optimizer, loss='mse')
plot_result(model, X, y_label, df)


"""1번 모델"""
## activation: sigmoid
## learning rate = 5.0
## Loss: 0.4055
model1_1 = layer_model('sigmoid', 5.0)
plot_result(model1_1, X, y_label, df)

## activation: sigmoid
## learning rate = 2.0
## Loss: 0.3993
model1_2 = layer_model('sigmoid', 2.0)
plot_result(model1_2, X, y_label, df)

## activation: sigmoid
## learning rate = 1.0
## Loss: 0.3992
model1_3 = layer_model('sigmoid', 1.0)
plot_result(model1_3, X, y_label, df)

## activation: sigmoid
## learning rate = 0.1
## Loss: 0.6155
model1_4 = layer_model('sigmoid', 0.1)
plot_result(model1_4, X, y_label, df)


"""2번 모델"""
## activation: tanh
## learning rate = 5.0
## Loss: 1.2036
model2_1 = layer_model('tanh', 5.0)
plot_result(model2_1, X, y_label, df)

## activation: tanh
## learning rate = 2.0
## Loss: 1.2036
model2_2 = layer_model('tanh', 2.0)
plot_result(model2_2, X, y_label, df)


## activation: tanh
## learning rate = 1.0
## Loss: 0.3705
model2_3 = layer_model('tanh', 1.0)
plot_result(model2_3, X, y_label, df)


## activation: tanh
## learning rate = 0.1
## Loss: 0.2894
model2_4 = layer_model('tanh', 0.1)
plot_result(model2_4, X, y_label, df)


"""
1) 기존 모델의 구조를 한층 늘리고 units수를 줄였지만 loss는 오히려 더 커졌다.
    layer를 늘릴 수록 꼭 loss가 줄어드는것이 아닌것을 알 수 있었다. 또한 unit의 수도 loss에 영향을 미친다는 것을 알 수 있었다.
    최종 손실값: 0.4056

2) activation 함수를 tanh로 바꾸고 learning rate를 0.1로 변경하였다.
    최종 손실값은 0.2894로 기존의 모델보다 감소하였다.

3) 1번과 2번에서 사용한 모델구조를 그대로 사용하고 학습률만 바꾸어 손실값을 구했다.
    model1: learning rate = 5.0 -> Loss: 0.4055
            learning rate = 2.0 -> Loss: 0.3993
            learning rate = 1.0 -> Loss: 0.3992
            learning rate = 0.1 -> Loss: 0.6155

    model2: learning rate = 5.0 -> Loss: 1.2036
            learning rate = 2.0 -> Loss: 1.2036
            learning rate = 1.0 -> Loss: 0.3705
            learning rate = 0.1 -> Loss: 0.2894
    
    최종 손실값은 activation 함수, learning rate, layer의 수, unit의 수와 같이 여러 변수들에 따라 변화하였다.
    모델에 따라 딱 맞는 값들이 존재하지 않고 직접 돌려가며 손실을 확인해야 한다는 것을 알 수 있었다.
    가장 적은 loss값을 가지는 변수를 찾는것이 중요하다는 것을 느꼈다.
"""