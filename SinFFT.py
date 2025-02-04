import numpy as np
import matplotlib.pyplot as plt

def aplicar_fourier_completo_modulo(signal, sample_rate):
    """
    Aplica la Transformada de Fourier a una señal y retorna las frecuencias y su magnitud
    calculando el módulo complejo explícitamente.

    Parámetros:
        signal (np.array): La señal de entrada en el dominio del tiempo.
        sample_rate (float): La tasa de muestreo de la señal (número de muestras por unidad de tiempo).

    Retorna:
        freqs (np.array): Las frecuencias correspondientes (positivas y negativas).
        magnitudes (np.array): La magnitud de las frecuencias calculada con el módulo complejo.
    """
    # Número de puntos en la señal
    n = len(signal)
    
    # Transformada de Fourier
    fft_values = np.fft.fft(signal)
    
    # Magnitud utilizando el módulo complejo: sqrt(real^2 + imag^2)
    magnitudes = np.sqrt(np.real(fft_values)**2 + np.imag(fft_values)**2) / n
    
    # Frecuencias asociadas (incluyendo negativas)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    
    return freqs[:n // 2], magnitudes[:n // 2] #Para positivas
    
    #return freqs, magnitudes #Para negativas y positivas


# Generar la señal de ejemplo
x = np.linspace(0, 4 * np.pi, 1000)  # 1000 puntos entre 0 y 4π
sample_rate = len(x) / (4 * np.pi)  # Tasa de muestreo aproximada
y = 5 * np.sin(x)  # Señal limpia con amplitud de 5
ruido = 0.3 * np.max(np.abs(y)) * np.random.normal(size=len(x))  # Ruido gaussiano
y_con_ruido = y + ruido

# Aplicar la Transformada de Fourier a la señal con ruido
freqs, magnitudes = aplicar_fourier_completo_modulo(y_con_ruido, sample_rate)

# Ordenar frecuencias para visualización (FFT produce salida en orden especial)
orden = np.argsort(freqs)
freqs = freqs[orden]
magnitudes = magnitudes[orden]

# Crear las gráficas
plt.figure(figsize=(15, 8))

# Gráfica de la señal limpia
plt.subplot(2, 2, 1)
plt.plot(x, y, label="Señal limpia (y = 5 sin(x))", color='b')
plt.title("Señal Limpia en el Dominio del Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Gráfica de la señal con ruido
plt.subplot(2, 2, 2)
plt.plot(x, y_con_ruido, label="Señal con ruido", color='r', alpha=0.7)
plt.title("Señal con Ruido en el Dominio del Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Gráfica del espectro de frecuencias de la señal limpia
plt.subplot(2, 1, 2)
plt.plot(freqs, magnitudes, label="Espectro de frecuencia (señal con ruido)", color='g', linewidth=1.5)
plt.title("Espectro de Frecuencia (Incluye Negativas)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
