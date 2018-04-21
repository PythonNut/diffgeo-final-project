# l inifity norm to find the max 
# wavelet transform or moving fourier window transform
# 
# draw the gyroscopic data wx wy wz
# the calculate the omega matrices and then exponentiate

import csv
import sys
import operator
import matplotlib.pyplot as plt
import numpy
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

filename_in = "public_dataset/100669/100669_session_1/Sorted_ID_12.csv"
file_in = open(filename_in, "r")
csvreader = csv.reader(file_in)

times = []
accel_x = []
accel_y = []
accel_z = []
i = 0

for row in csvreader:
	if i < 10:
		times.append(int(row[0]))
		accel_x.append(float(row[3]))
		accel_y.append(float(row[4]))
		accel_z.append(float(row[5]))
		i = i+1
	else:
		i = i+1
		continue

# computes the difference of the arrays
times_diff = numpy.diff(times)
accel_x_diff = numpy.diff(accel_x)
accel_y_diff = numpy.diff(accel_y)
accel_z_diff = numpy.diff(accel_z)

# computes the numerical derivative of each of the acceleration curves
accel_x_deriv = accel_x_diff/times_diff
accel_y_deriv = accel_y_diff/times_diff
accel_z_deriv = accel_z_diff/times_diff

# computes second order difference of arrays
times_diff2 = times_diff[0:len(times_diff)-1]
accel_x_diff2 = numpy.diff(accel_x_diff)
accel_y_diff2 = numpy.diff(accel_y_diff)
accel_z_diff2 = numpy.diff(accel_z_diff)

# computes the second derivate of the acceleration
accel_x_deriv2 = accel_x_diff2/times_diff2
accel_y_deriv2 = accel_y_diff2/times_diff2
accel_z_deriv2 = accel_z_diff2/times_diff2

# computes third order difference of arrays
times_diff3 = times_diff2[0:len(times_diff2)-1]
accel_x_diff3 = numpy.diff(accel_x_diff2)
accel_y_diff3 = numpy.diff(accel_y_diff2)
accel_z_diff3 = numpy.diff(accel_z_diff2)

# computes the third derivate of the acceleration
accel_x_deriv3 = accel_x_diff3/times_diff3
accel_y_deriv3 = accel_y_diff3/times_diff3
accel_z_deriv3 = accel_z_diff3/times_diff3

curvature = []
torsion = []

# computes the curvature and torsion
for i in range(0, len(accel_x_deriv2)):
	accel_deriv = [accel_x_deriv[i], accel_y_deriv[i], accel_z_deriv[i]]
	curv_denom = numpy.linalg.norm(accel_deriv)**3
	accel_deriv2 = [accel_x_deriv2[i], accel_y_deriv2[i], accel_z_deriv2[i]]
	cross_prod = numpy.cross(accel_deriv, accel_deriv2)
	cross_prod_mag = numpy.linalg.norm(cross_prod)

	curvature.append(cross_prod_mag/curv_denom)

# compute the torsion
for i in range(0, len(accel_z_deriv3)):
	accel_deriv = [accel_x_deriv[i], accel_y_deriv[i], accel_z_deriv[i]]
	accel_deriv2 = [accel_x_deriv2[i], accel_y_deriv2[i], accel_z_deriv2[i]]
	accel_deriv3 = [accel_x_deriv3[i], accel_y_deriv3[i], accel_z_deriv3[i]]

	cross_prod_tor = numpy.cross(accel_deriv, accel_deriv2)
	num_tor = numpy.dot(cross_prod_tor, accel_deriv3)
	denom_tor = numpy.linalg.norm(cross_prod_tor)**2

	torsion.append(num_tor/denom_tor)


# formatting for new times to be the same size as the derivative 
times2 = times[0:len(times)-1]
times3 = times2[0:len(times2)-1]
times4 = times3[0:len(times3)-1]

# velocities of each direction given by integrating the acceleration with
# a cumulative trapezoid method
veloc_x = integrate.cumtrapz(accel_x, times)
veloc_y = integrate.cumtrapz(accel_y, times)
veloc_z = integrate.cumtrapz(accel_z, times)

pos_x = integrate.cumtrapz(veloc_x, times2)
pos_y = integrate.cumtrapz(veloc_y, times2)
pos_z = integrate.cumtrapz(veloc_z, times2)

# plt.figure(1)
# plt.plot(times, accel_x)
# plt.title("Acceleration x")

# plt.figure(2)
# plt.plot(times2, veloc_x)
# plt.title("Velocity x")

# plt.figure(3)
# plt.plot(times3, pos_x)
# plt.title("Position x")

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(times, accel_x)
plt.ylabel('acceleration x')
plt.subplot(2, 1, 2)
plt.plot(times2, accel_x_deriv)
plt.xlabel('time')
plt.ylabel('jerk x')
plt.savefig('images/accel_jerk_x.png')

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(times, accel_y)
plt.ylabel('acceleration y')
plt.subplot(2, 1, 2)
plt.plot(times2, accel_y_deriv)
plt.xlabel('time')
plt.ylabel('jerk y')
plt.savefig('images/accel_jerk_y.png')

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(times, accel_z)
plt.ylabel('acceleration z')
plt.subplot(2, 1, 2)
plt.plot(times2, accel_z_deriv)
plt.xlabel('time')
plt.ylabel('jerk z')
plt.savefig('images/accel_jerk_z.png')

plt.figure(4)
plt.plot(times3, curvature)
plt.ylabel('curvature')
plt.xlabel('time')
plt.savefig('images/curvature-12.png')

plt.figure(5)
plt.plot(times4, torsion)
plt.ylabel('torsion')
plt.xlabel('time')
plt.savefig('images/torsion-12.png')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(accel_x,accel_y,accel_z)
plt.title('acceleration')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(pos_x,pos_y,pos_z)
plt.title('position')

print("Size times: " + str(len(times)))
print("Size x: " + str(len(accel_x)))
print("Size y: " + str(len(accel_y)))
print("Size z: " + str(len(accel_z)))

plt.show()