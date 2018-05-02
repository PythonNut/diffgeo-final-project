import csv
import numpy as np
from numpy.random import randn
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt

file = "public_dataset/100669/100669_session_9/Accelerometer.csv"
file_in = open(file, "r")
csvreader = csv.reader(file_in)

T = []
Xa = []
Ya = []
Za = []

for row in csvreader:
	T.append(int(row[0]))
	Xa.append(float(row[3]))
	Ya.append(float(row[4]))
	Za.append(float(row[5]))