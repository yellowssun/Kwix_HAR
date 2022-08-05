import os
import numpy as np

babel_actions = ['over_head_press', 'side_raise', 'babel_curl', 'deadlift', 'squart']
body_actions = ['crunch', 'knee_up', 'leg_raise', 'side_crunch', 'side_lunge']

for action in body_actions:
    DATA_PATH = 'E:/fitness_image/Training'
    new_dir = 'E:/fitness_image/Training' + '/' + action
    pose_list = 'C'
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
    sequence = 26


    if action == 'knee_up' or 'side_crunch':
        sequence = 25
        print(action, sequence)
    elif action == 'side_lunge':
        sequence = 26
        print(action, sequence)
    else:
        sequence = 21
        print(action, sequence)

    for i in range(1, sequence+1):
            os.makedirs(os.path.join(new_dir, str(i), pose_list), exist_ok=True)
