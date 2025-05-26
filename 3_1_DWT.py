# Practica 9: Transformada Wavelet Discreta (DWT) en Python con NumPy.
# Procesamiento Digital de señales
# Autor: Mario Yahir Garcia Hernandez

# Objetivo: Implementar manualmente la DWT usando el wavelet de Haar para analizar una señal
# unidimensional.

# a) Grafique la señal de aproximación (A) y compárela con la señal original
# b) Extender a múltiples niveles: Modifica el código para aplicar la DWT en cascada (usar A como
# nueva señal de entrada) y extienda a 4 niveles. Muestre los niveles de detalle en una gráfica
# c) Investigue como reconstruir la señal y compárela con la original, (HINT: resuelva para 𝑥2𝑘, 𝑥2𝑘+1
# en cada uno de los 4 niveles solicitados). Use un humbral de 0.5 para eliminar coeficientes
# pequeños

import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Señales y procesamiento ===

def generate_signals(t):
    clean_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    noise = 0.5 * np.random.normal(0, 1, len(t))
    noisy_signal = clean_signal + noise
    return clean_signal, noisy_signal

def haar_dwt(signal):
    A = (signal[::2] + signal[1::2]) / np.sqrt(2)
    D = (signal[::2] - signal[1::2]) / np.sqrt(2)
    return A, D

def haar_idwt(A, D):
    x_even = (A + D) / np.sqrt(2)
    x_odd  = (A - D) / np.sqrt(2)
    result = np.empty(x_even.size + x_odd.size)
    result[0::2], result[1::2] = x_even, x_odd
    return result

def multi_level_dwt(signal, levels):
    coeffs = []
    current = signal
    for _ in range(levels):
        A, D = haar_dwt(current)
        coeffs.append(D)
        current = A
    coeffs.append(current)  # última aproximación
    return coeffs

def reconstruct_with_threshold(coeffs, threshold=0.5):
    A = coeffs[-1]
    for D in reversed(coeffs[:-1]):
        D[np.abs(D) < threshold] = 0
        A = haar_idwt(A, D)
    return A

# === Configuración principal ===

np.random.seed(42)
N = 512
t = np.linspace(0, 1, N, endpoint=False)
clean_signal, noisy_signal = generate_signals(t)

# DWT y reconstrucción
coeffs = multi_level_dwt(noisy_signal, levels=4)
reconstructed = reconstruct_with_threshold(coeffs.copy(), threshold=0.5)
approx_final = coeffs[-1]
error = clean_signal - reconstructed

# === GUI con customtkinter ===

ctk.set_appearance_mode("light")
app = ctk.CTk()
app.title("DWT Haar - Visualización Interactiva")
app.geometry("1100x750")

tabview = ctk.CTkTabview(app)
tabview.pack(expand=True, fill="both")

# === Tab a) Señal original y aproximación nivel 1 ===
tab_a = tabview.add("a) Señal vs Aproximación")

fig_a, ax_a = plt.subplots(figsize=(10, 4))
ax_a.plot(t, noisy_signal, label="Señal con ruido", alpha=0.6)
ax_a.plot(np.linspace(0, 1, len(coeffs[0])), coeffs[0], label="Aproximación Nivel 1", color='green')
ax_a.set_title("Señal con Ruido vs Aproximación Nivel 1")
ax_a.legend()
ax_a.grid(True)

canvas_a = FigureCanvasTkAgg(fig_a, master=tab_a)
canvas_a.get_tk_widget().pack(fill="both", expand=True)
canvas_a.draw()

# === Tab b) Detalles por nivel ===
tab_b = tabview.add("b) Detalles Wavelet")

fig_b, axs_b = plt.subplots(4, 1, figsize=(10, 8))
for i in range(4):
    axs_b[i].plot(coeffs[i])
    axs_b[i].set_title(f"Detalle Nivel {i+1}")
    axs_b[i].grid(True)

fig_b.tight_layout()
canvas_b = FigureCanvasTkAgg(fig_b, master=tab_b)
canvas_b.get_tk_widget().pack(fill="both", expand=True)
canvas_b.draw()

# === Tab c) Reconstrucción con umbral ===
tab_c = tabview.add("c) Reconstrucción")

fig_c, ax_c = plt.subplots(figsize=(10, 4))
ax_c.plot(t, clean_signal, label="Señal original", alpha=0.4)
ax_c.plot(t, reconstructed, label="Señal reconstruida (umbral 0.5)", color='red')
ax_c.set_title("Reconstrucción de Señal con Umbralización")
ax_c.legend()
ax_c.grid(True)

canvas_c = FigureCanvasTkAgg(fig_c, master=tab_c)
canvas_c.get_tk_widget().pack(fill="both", expand=True)
canvas_c.draw()

# === Tab d) Error de reconstrucción ===
tab_d = tabview.add("Error de Reconstrucción")

fig_d, ax_d = plt.subplots(figsize=(10, 4))
ax_d.plot(t, error, label="Error (original - reconstruida)", color='black')
ax_d.set_title("Error de Reconstrucción")
ax_d.legend()
ax_d.grid(True)

canvas_d = FigureCanvasTkAgg(fig_d, master=tab_d)
canvas_d.get_tk_widget().pack(fill="both", expand=True)
canvas_d.draw()

app.mainloop()
