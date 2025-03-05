import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def bode_plot_lowpass(fc=1000, fs=10000):
    # Definir la frecuencia de corte y la frecuencia de muestreo
    w0 = 2 * np.pi * fc  # Frecuencia de corte en rad/s
    
    # Coeficientes del filtro pasa-bajas de primer orden
    b, a = signal.butter(1, w0, btype='low', analog=True)
    
    # Generar la respuesta en frecuencia
    w, mag, phase = signal.bode((b, a))
    
    # Graficar la respuesta en magnitud
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title("Diagrama de Bode - Magnitud")
    plt.xlabel("Frecuencia (rad/s)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    
    # Graficar la respuesta en fase
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.title("Diagrama de Bode - Fase")
    plt.xlabel("Frecuencia (rad/s)")
    plt.ylabel("Fase (grados)")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# Llamar a la función con frecuencia de corte de 1 kHz
bode_plot_lowpass(1000)
