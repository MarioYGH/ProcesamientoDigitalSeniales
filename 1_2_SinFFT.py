# Practica 1: Ruido y FFT
# Procesamiento Digital de señales
# Autor: Mario Yahir Garcia Hernandez

 # Aplica la Transformada de Fourier a una señal y retorna las frecuencias y su magnitud
 # calculando el módulo complejo explícitamente.

 # Parámetros:
 #     signal (np.array): La señal de entrada en el dominio del tiempo.
 #     sample_rate (float): La tasa de muestreo de la señal.

 # Retorna:
 #     freqs (np.array): Frecuencias positivas.
 #     magnitudes (np.array): Magnitud de las frecuencias.

import numpy as np
import matplotlib.pyplot as plt

def aplicar_fourier_completo_modulo(signal, sample_rate):
    """
    Aplica la Transformada de Fourier a una señal y retorna las frecuencias y su magnitud
    calculando el módulo complejo explícitamente.

    Parámetros:
        signal (np.array): La señal de entrada en el dominio del tiempo.
        sample_rate (float): La tasa de muestreo de la señal.

    Retorna:
        freqs (np.array): Frecuencias positivas.
        magnitudes (np.array): Magnitud de las frecuencias.
    """
    # Número de puntos en la señal
    n = len(signal)
    
    # Transformada de Fourier
    fft_values = np.fft.fft(signal)
    
    # Magnitud utilizando el módulo complejo
    magnitudes = np.sqrt(np.real(fft_values)**2 + np.imag(fft_values)**2) / n
    
    # Frecuencias asociadas
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    
    return freqs[:n // 2], magnitudes[:n // 2]  # Solo frecuencias positivas


# Generar la señal de ejemplo
x = np.linspace(0, 4 * np.pi, 1000)  
sample_rate = len(x) / (4 * np.pi)  
y = 5 * np.sin(x)  
ruido = 0.3 * np.max(np.abs(y)) * np.random.normal(size=len(x))  
y_con_ruido = y + ruido

# Aplicar la Transformada de Fourier
freqs, magnitudes = aplicar_fourier_completo_modulo(y_con_ruido, sample_rate)

# Ordenar las frecuencias
orden = np.argsort(freqs)
freqs = freqs[orden]
magnitudes = magnitudes[orden]

# Mostrar información clave en consola
print("\n===== Información de la Señal =====")
print(f"Número de muestras: {len(x)}")
print(f"Tasa de muestreo: {sample_rate:.4f} Hz")

print("\n===== Información de la Transformada de Fourier =====")
print(f"Número de frecuencias analizadas: {len(freqs)}")
print(f"Frecuencia con mayor magnitud: {freqs[np.argmax(magnitudes)]:.4f} Hz")
print(f"Magnitud máxima en el espectro: {np.max(magnitudes):.4f}")

# Gráficas
plt.figure(figsize=(15, 8))

# Señal limpia
plt.subplot(2, 2, 1)
plt.plot(x, y, label="Señal limpia (y = 5 sin(x))", color='b')
plt.title("Señal Limpia en el Dominio del Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Señal con ruido
plt.subplot(2, 2, 2)
plt.plot(x, y_con_ruido, label="Señal con ruido", color='r', alpha=0.7)
plt.title("Señal con Ruido en el Dominio del Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Espectro de frecuencias
plt.subplot(2, 1, 2)
plt.plot(freqs, magnitudes, label="Espectro de frecuencia (señal con ruido)", color='g', linewidth=1.5)
plt.title("Espectro de Frecuencia (Frecuencias Positivas)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
