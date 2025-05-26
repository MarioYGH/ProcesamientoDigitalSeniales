# Practica 8: Transformada discreta de Fourier 
# Procesamiento Digital de señales
# Autor: Mario Yahir Garcia Hernandez

# Objetivo: Implementar la Transformada Discreta de Fourier (DFT) desde cero utilizando NumPy, sin recurrir a
# funciones especializadas como numpy.fft.fft. Analizar su aplicación en señales discretas y comparar los
# resultados con la implementación estándar.

import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Implementación manual de la DFT
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Implementación manual de la IDFT
def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

# Crear señal de 4 gaussianas
def generate_signal():
    x = np.zeros(2048)
    x += 1.0 * np.exp(-0.5 * ((np.arange(2048) - 500) / 30)**2)
    x += 0.6 * np.exp(-0.5 * ((np.arange(2048) - 800) / 25)**2)
    x += 0.3 * np.exp(-0.5 * ((np.arange(2048) - 1200) / 40)**2)
    x += 0.8 * np.exp(-0.5 * ((np.arange(2048) - 1600) / 20)**2)
    return x

# Filtrado por frecuencia
def frequency_filter(X, fs, threshold):
    N = len(X)
    freqs = np.fft.fftfreq(N, d=1/fs)
    X_filtered = np.copy(X)
    X_filtered[np.abs(freqs) > threshold] = 0
    return X_filtered

# Métricas
def mse(x1, x2):
    return np.mean((x1 - x2)**2)

def snr(signal, noise):
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise)

# Actualizar gráfica
def update():
    original = generate_signal()
    noise = np.random.normal(0, 0.1, size=original.shape)
    noisy = original + noise

    X = DFT(noisy)
    X_filtered = frequency_filter(X, fs=1.0, threshold=0.025)
    x_filtered = IDFT(X_filtered).real

    ax[0].clear()
    ax[0].plot(original, label="Original")
    ax[0].plot(noisy, label="Con ruido", alpha=0.6)
    ax[0].legend()
    ax[0].set_title("Señal con ruido")

    ax[1].clear()
    ax[1].plot(np.abs(X), label="DFT")
    ax[1].set_title("Transformada Discreta de Fourier")

    ax[2].clear()
    ax[2].plot(x_filtered, label="Filtrada")
    ax[2].legend()
    ax[2].set_title("Señal Filtrada")

    canvas.draw()

    # Imprimir respuestas
    print("\nAnálisis de la DFT y filtrado:")
    print(f"1. MSE entre señal original y filtrada: {mse(original, x_filtered):.6f}")
    print(f"2. SNR de la señal ruidosa: {snr(original, noise):.2f} dB")
    print("3. Se observa que el filtrado atenúa componentes de alta frecuencia (>0.025 Hz).")

# Interfaz
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Transformada Discreta de Fourier (DFT)")
root.geometry("900x800")

ctk.CTkLabel(root, text="Implementación de la DFT Manual", font=("Arial", 16)).pack(pady=10)

btn = ctk.CTkButton(root, text="Ejecutar Transformada y Filtrado", command=update)
btn.pack(pady=10)

fig, ax = plt.subplots(3, 1, figsize=(9, 8))
plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True)

update()
root.mainloop()
