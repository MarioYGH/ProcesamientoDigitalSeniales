import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
global ax

# Definir la función periódica cuadrada
def square_wave(t, T):
    return np.where(t < T/2, 1, -1)

# Calcular coeficientes de Fourier con regla del trapecio
def compute_fourier_coeffs(N, T, t, f_t):
    dt = t[1] - t[0]
    a0 = (2 / T) * np.trapz(f_t, dx=dt)
    an = np.array([(2 / T) * np.trapz(f_t * np.cos(n * 2 * np.pi * t / T), dx=dt) for n in range(1, N+1)])
    bn = np.array([(2 / T) * np.trapz(f_t * np.sin(n * 2 * np.pi * t / T), dx=dt) for n in range(1, N+1)])
    return a0, an, bn

# Reconstrucción de la Serie de Fourier
def fourier_series(N, T, t, a0, an, bn):
    f_approx = a0 / 2 + sum(an[n-1] * np.cos(n * 2 * np.pi * t / T) + bn[n-1] * np.sin(n * 2 * np.pi * t / T) for n in range(1, N+1))
    return f_approx

# Función para actualizar la gráfica
def update_plot(event=None):
    N = int(slider.get())
    a0, an, bn = compute_fourier_coeffs(N, T, t, f_t)
    f_approx = fourier_series(N, T, t, a0, an, bn)

    ax.clear()
    ax.plot(t, f_t, label='Onda Cuadrada', linestyle='dashed')
    ax.plot(t, f_approx, label=f'Serie de Fourier (N={N})')
    ax.legend()
    ax.set_title("Aproximación de Serie de Fourier")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Amplitud")
    canvas.draw()

    # Responder preguntas en la terminal
    print(f"\nCon N={N}:")
    print("1. Efecto de N: A mayor N, la aproximación mejora, pero aparece el fenómeno de Gibbs en los bordes.")
    print(f"2. Coeficientes an cercanos a cero: Se debe a la simetría impar de la función, lo que anula los términos pares.")
    print("3. Se necesitan muchos términos porque la onda cuadrada tiene discontinuidades y los senos y cosenos son suaves.")

# Configuración de la ventana
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Aproximación de Serie de Fourier")
root.geometry("900x700")

# Parámetros iniciales
T = 2 * np.pi
t = np.linspace(0, T, 1000)
f_t = square_wave(t, T)

# Crear interfaz
title = ctk.CTkLabel(root, text="Aproximación de Serie de Fourier", font=("Arial", 16))
title.pack(pady=10)

slider = ctk.CTkSlider(root, from_=1, to=50, number_of_steps=49)
slider.set(10)
slider.pack(pady=10)
slider.bind("<B1-Motion>", update_plot)
slider.bind("<ButtonRelease-1>", update_plot)

fig, ax = plt.subplots(figsize=(7, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10, expand=True, fill='both')

# Primera actualización
update_plot()

root.mainloop()
