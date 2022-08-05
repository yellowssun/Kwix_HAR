import os
import shutil
import numpy as np
import re

regex = r"\d\d\d-\d-\d-21-\S\d{1,3}_\S"
path_dir = 'E:/fitness_image/Training'
angle_list = 'C'
num_person = 1

day_list1 = np.array(['Day01_200921_F', 'Day05_200925_F', 'Day23_201024_F', 'Day27_201029_F']) ## 종목: 1~5
day_list2 = np.array(['Day02_200922_F', 'Day24_201026_F', 'Day28_201030_F'])                   ## 종목: 5~12
day_list3 = np.array(['Day33_201105_F', 'Day35_201107_F', 'Day36_201108_F'])                   ## 종목: 21~29
day_list4 = np.array(['Day08_201005_F', 'Day12_201012_F', 'Day30_201102_F'])                   ## 종목: 17~21
day_list5 = np.array(['Day11_201008_F', 'Day15_201015_F', 'Day25_201027_F', 'Day29_201031_F']) ## 종목: 12~16


for idx, Day in enumerate(day_list3):
    path = 'E:/fitness_image/Training' + '/' + day_list3[idx]
    print(day_list3[idx])

    if day_list3[idx] == 'Day23_201024_F' or  day_list3[idx] == 'Day24_201026_F' or day_list3[idx] == 'Day28_201030_F' or day_list3[idx] == 'Day27_201029_F':
        n_person = 9

    elif day_list3[idx] == 'Day33_201105_F' or day_list3[idx] == 'Day35_201107_F' or day_list3[idx] == 'Day36_201108_F':
        n_person = 7

    else:
        n_person = 8



    for person in range(1, n_person+1):
        path1 = path + '/' + str(person)             ## folder_dir -> D:\피트니스 자세 이미지\Training\Day01_200921_F\1
        print(path1)

        for angle in angle_list:
            path2 = path1 + '/' + str(angle)          ## folder_dir -> D:\피트니스 자세 이미지\Training\Day01_200921_F\1\A
            folder_list = os.listdir(path2)

            new_path = 'E:/fitness_image/Training/leg_raise'
            new_path = new_path + '/' + str(num_person)      ## D:\피트니스 자세 이미지\Training\standing_knee_up\1
            new_path = new_path + '/' + str(angle)           ## D:\피트니스 자세 이미지\Training\standing_knee_up\1\A
            print(new_path)

            matches = re.findall(regex, str(folder_list), re.MULTILINE)

            for match in matches: 
                path3 = path2 + '/' + match    ## folder_dir -> D:\피트니스 자세 이미지\Training\Day01_200921_F\1\C\033-1-1-02-Z2_A
                file_list = os.listdir(path3)

                for i in file_list:
                    shutil.copy(path3 + '/' + i, new_path)

        num_person += 1
