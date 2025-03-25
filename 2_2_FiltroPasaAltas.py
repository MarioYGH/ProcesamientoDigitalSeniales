import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fc = 10  # Frecuencia de corte (Hz)
fs = 1000  # Frecuencia de muestreo (Hz)
dt = 1 / fs  # Intervalo de tiempo
tau = 1 / (2 * np.pi * fc)  # Constante de tiempo
alpha = tau / (tau + dt)  # Coeficiente del filtro

# Crear el vector de tiempo y la señal de entrada
t = np.arange(0, 1, dt)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)  # Señal compuesta

# Inicializar la señal filtrada
y = np.zeros_like(x)

# Implementación del filtro pasa-altas usando la ecuación recursiva
for n in range(1, len(x)):
    y[n] = alpha * (y[n-1] + x[n] - x[n-1])

# Graficar la señal de entrada y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Señal de entrada')
plt.title("Señal de entrada y señal filtrada (dominio del tiempo)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, y, label='Señal filtrada', color='orange')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()

# Transformada de Fourier de ambas señales
X = np.fft.fft(x)
Y = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(t), dt)

plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(X)[:len(X)//2], label='Espectro de x(t)')
plt.plot(frequencies[:len(frequencies)//2], np.abs(Y)[:len(Y)//2], label='Espectro de y(t)', color='orange')
plt.title("Espectro de ambas señales")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.legend()
plt.grid()
plt.show()

print("¿Cómo afecta la frecuencia de corte 𝑓𝑐 a la señal filtrada? Si 𝑓𝑐 es más alta, el filtro atenúa menos las frecuencias altas. Si es más baja, eliminará más componentes altas.")
print("¿Qué ocurre si aumentas o disminuyes la frecuencia de muestreo 𝑓𝑠? Si 𝑓𝑠 disminuye, la resolución temporal es menor, lo que puede causar aliasing. Si 𝑓𝑠 aumenta, el filtro opera mejor con mayor precisión.")
print("¿Cómo se compara este filtro con un filtro pasa-altas ideal en el dominio de la frecuencia? Un filtro ideal elimina completamente las frecuencias bajas. Este filtro atenúa gradualmente las bajas, debido a su comportamiento de primer orden.")
