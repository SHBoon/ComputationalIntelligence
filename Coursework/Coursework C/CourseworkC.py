import scipy.io as spio

mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

d = mat['d']

Index = mat['Index']

Class = mat['Class']