class Button1():
    def __init__(self,pos,text, size=[50,50]):
        self.pos = pos
        self.size = size
        self.text = text


class Button2():
    def __init__(self,pos,text, size=[450,50]):
        self.pos = pos
        self.size = size
        self.text = text
        

# Name
alpha_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Height, Weight, Old
num_keys = [["7", "8", "9"],
            ["4", "5", "6"],
            ["1", "2", "3"],
            ["*", "0", "#"]]

# Sex
sex_keys = [["Male"],
            ["Female"]]

# Exercise mode
ex_keys = [["AI-PT"],
           ["Free exercise"]]

buttonList1 = []
buttonList2 =[]
buttonList3 = []
buttonList4 = []

for i in range(len(alpha_keys)):
    for x, key in enumerate(alpha_keys[i]):
        buttonList1.append(Button1([100*x+40, 100*i +40], key))
        
for i in range(len(num_keys)):
    for x, key in enumerate(num_keys[i]):
        buttonList2.append(Button1([100*x+40, 100*i +40], key))
        
for i in range(len(sex_keys)):
    for x, key in enumerate(sex_keys[i]):
        buttonList3.append(Button2([100*x+40, 100*i +40], key))
        
for i in range(len(ex_keys)):
    for x, key in enumerate(ex_keys[i]):
        buttonList4.append(Button2([100*x+40, 100*i +40], key))