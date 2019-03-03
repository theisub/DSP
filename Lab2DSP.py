import numpy as np
import matplotlib.pyplot as plt
#DFT
def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

    
#инициализация начальных параметров
T = 5
step = 0.05
N = 20 # число отсчетов
dt = T / (N-1)
sigma = 0.5


#инициализация для прямоугольного импульса
x_rect = np.arange(-T,T,step) 
y_rect = np.where(abs(x_rect)<=abs(T/2),1,0)


#инициализация для гауссового импульса
x_gauss = np.arange(-T,T,step)
y_gauss = np.exp(-x_gauss ** 2/ sigma ** 2)


# гауссовый fft
у_gauss_fft = np.fft.fft(y_gauss)
y_gauss_shift_fft = np.fft.fftshift(у_gauss_fft)

plt.plot(x_gauss,abs(у_gauss_fft))
plt.plot(x_gauss,abs(y_gauss_shift_fft))

plt.show()


# прямоугольный fft
y_rect_fft = np.fft.fft(y_rect)
y_rect_shift_fft = np.fft.fftshift(y_rect_fft)

plt.plot(x_rect,abs(y_rect_fft))
plt.plot(x_rect,abs(y_rect_shift_fft))

plt.show()

# прямоугольный  dft
y_rect_dft = DFT(y_rect)
y_rect_shift_dft = np.fft.fftshift(y_rect_dft)

plt.plot(x_rect,abs(y_rect_dft))
plt.plot(x_rect,abs(y_rect_shift_dft))

plt.show()

# гауссовый dft
y_gauss_dft = DFT(y_gauss)
y_gauss_shift_dft = np.fft.fftshift(y_gauss_dft)

plt.plot(x_gauss,abs(y_gauss_dft))
plt.plot(x_gauss,abs(y_gauss_shift_dft))

plt.show()



