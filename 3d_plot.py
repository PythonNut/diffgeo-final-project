import csv
import sys
import operator
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# definitions for all file paths for relevant user and session
user = "100669"
session = "1"
filepath = "public_dataset/" + user + "/" + user + "_session_" + session + "/"
csv_accel = filepath + "Accelerometer.csv"
csv_mag = filepath + "Magnetometer.csv"
csv_gyro = filepath + "Gyroscrope.csv"
csv_touch = filepath + "TouchEvent.csv"
csv_keypress = filepath + "KeyPressEvent.csv"
csv_onefinger = filepath + "OneFingerTouchEvent.csv"
csv_pinch = filepath + "PinchEvent.csv"
csv_scroll = filepath + "ScrollEvent.csv"
csv_stroke = filepath + "StrokeEvent.csv"

file_accel = open(csv_accel, "r")
csvreader_accel = csv.reader(file_accel)

times_abs = []
times_rel = []
accel_x = []
accel_y = []
accel_z = []

for row in csvreader_accel:
	times_abs.append(int(row[0]))
	times_rel.append(int(row[1]))
	accel_x.append(float(row[3]))
	accel_y.append(float(row[4]))
	accel_z.append(float(row[5]))

fig = plt.figure()
ax = fig.gca(projection='3d')
# line, = ax.plot(accel_x, accel_y, accel_z, label='parametric curve')

def gen(n):
	i = 0
	while i < n:
		yield np.array([accel_x[i],accel_y[i],accel_z[i]])
		i = i+1

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

N = 10000
data = np.array(list(gen(N))).T
#line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
line, = ax.plot(accel_x, accel_y, accel_z, label='parametric curve')
ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=40, blit=False)
#ani.save('matplot003.gif', writer='imagemagick')
plt.show()