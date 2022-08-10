import numpy as np
import tensorflow as tf

x = np.random.randint(100, size=(500, 15, 47))
print('x shape:', x.shape)
print('pre x shape:', x.shape[1:3])
_x = np.random.randint(100, size=(15, 47))
print('_x shape:', _x.shape)

y = np.expand_dims(_x, axis=2)
print('y shape:', y.shape)