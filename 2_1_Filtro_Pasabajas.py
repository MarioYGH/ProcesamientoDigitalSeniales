# Practica 5: Filtro Pasa-Bajas en Diferencias Finitas
# Procesamiento Digital de señales
# Autor: Mario Yahir Garcia Hernandez

# Objetivo: Implementar un filtro pasa-bajas en python utilizando el metodo de 
# diferencias finitas. Analizar la respuesta del filtro y su comportamiento  
# en el dominio de la frecuencia  

# a) Sustituye en la ecuación diferencial y despeja para y[n] para asi obtener la ecuación recursiva para implementar en Python 
# Considerando la señal de entrada: 
# 1 x = sin(360°.5t) + sin(360°-50t) 
# b) Implementa un filtro pasa baja (ecuación recursiva de arriba) en la señal de entrada, considere 
# los siguientes parámetros 
# a. Frecuencia de corte f = 10hz 
# b. Frecuencia de muestreo f = 1000hz 
# c. Intervalo de tiempo At = fs 
# c) Visualice los resultados 
# a. Graficar la señal de entrada y la señal filtrada en el dominio del tiempo 
# d) Análisis en el dominio de la frecuencia 
# a. Calcular la transformada de Fourier de la señal de entrada y la señal filtrada. 
# b. Graficar la magnitud del espectro de ambas señales.


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

# Pregunta a: 
print("¿Cómo afecta la frecuencia de corte (fc) a la señal filtrada? Si aumentas la frecuencia de corte (fc), el filtro pasa-bajas permitirá el paso de señales de mayor frecuencia, atenuando menos las altas frecuencias. Si disminuyes fc, el filtro bloqueará más componentes de alta frecuencia, dejando pasar principalmente las más bajas.")

# Pregunta b: 
print("¿Qué ocurre si aumentas o disminuyes la frecuencia de muestreo (fs)? Si aumentas la frecuencia de muestreo (fs), puedes representar mejor señales de alta frecuencia sin aliasing y el filtro será más preciso. Si fs disminuye, puedes perder información de señales de alta frecuencia y puede aparecer aliasing.")

# Pregunta c: 
print("¿Cómo se compara este filtro con un filtro pasa-bajas ideal en el dominio de la frecuencia? Este filtro pasa-bajas atenúa gradualmente las señales de alta frecuencia debido a su comportamiento de primer orden. Un filtro ideal eliminaría completamente todas las frecuencias por encima de la frecuencia de corte, mientras que este filtro tiene una transición más suave.")
