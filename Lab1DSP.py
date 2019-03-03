import numpy as np
import matplotlib.pyplot as plt

# Прямоугольный сигнал
T = 1
step = 0.01
x_rect = np.arange(-T,T,step) 
y_rect = np.where(abs(x_rect)<=abs(T/2),1,0)

plt.plot(x_rect,y_rect)

N = 20 # число отсчетов

dt = T / (N-1)
tl = np.arange(-T,T,dt)

y_rectrec = np.where(abs(tl)<=abs(T/2),1,0)

y_restored = np.zeros(len(y_rectrec))

for i in range(len(tl)):
    for j in range(len(y_rectrec)):
        k = i - round(len(y_rectrec)/2)
        y_restored[j] = y_restored[j] + y_rectrec[i] * np.sinc(np.pi/dt * (tl[j] - k * dt))
    
plt.plot(tl,y_restored)
plt.show()

#Сигнал Гаусса
sigma = 0.1

dt = T / (N-1)
x_gauss = np.arange(-T,T,dt)

y_gauss = np.exp(-x_gauss ** 2/ sigma ** 2)

plt.plot(x_gauss,y_gauss)

y_restored = np.zeros(len(x_gauss))

for i in range(len(x_gauss)):
    for j in range(len(y_gauss)):
        k = i - round(len(y_gauss)/2)
        y_restored[j] = y_restored[j] + y_gauss[i] * np.sinc(np.pi/dt * (x_gauss[j] - k * dt))

plt.plot(x_gauss,y_restored)
plt.show()
