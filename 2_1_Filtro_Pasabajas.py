import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Parámetros del filtro y señal
fc = 10  # Frecuencia de corte en Hz
fs = 1000  # Frecuencia de muestreo en Hz
dt = 1/fs  # Intervalo de muestreo
T = 1  # Duración de la señal en segundos
N = int(T * fs)  # Número de muestras

# Generación de la señal de entrada
t = np.linspace(0, T, N, endpoint=False)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# Parámetros del filtro
tau = 1 / (2 * np.pi * fc)
alfa = dt / (tau + dt)

# Aplicación del filtro pasa-bajas
y = np.zeros_like(x)
for n in range(1, N):
    y[n] = alfa * x[n] + (1 - alfa) * y[n - 1]

# Graficar señales en el dominio del tiempo
plt.figure(figsize=(10, 4))
plt.plot(t, x, label="Señal de entrada", alpha=0.6)
plt.plot(t, y, label="Señal filtrada", linewidth=2)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal antes y después del filtrado")
plt.legend()
plt.grid()
plt.show()

# Transformada de Fourier
X_f = fft(x)
Y_f = fft(y)
freqs = fftfreq(N, dt)

# Graficar el espectro de frecuencias
plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(X_f[:N//2]) / N, label="Entrada", alpha=0.6)
plt.plot(freqs[:N//2], np.abs(Y_f[:N//2]) / N, label="Filtrada", linewidth=2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("Espectro de frecuencia antes y después del filtrado")
plt.legend()
plt.grid()
plt.show()
