import time
from tracemalloc import start

start_time = time.time()

for i in range(100000000):
    i = 1
end_time = time.time()

predict_time = end_time - start_time
print(f'Excution time: {predict_time}')