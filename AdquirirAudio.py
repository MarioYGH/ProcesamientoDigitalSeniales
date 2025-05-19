import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración global
DURATION = 2  # segundos por palabra
FS = 44100    # frecuencia de muestreo

# Datos
audios = []       # Señales originales (tiempo)
fft_data = []     # Magnitud del espectro (frecuencia)

# Función para grabar audio
def grabar_y_guardar(palabra_n):
    global audio4_fft
    if palabra_n < 3:
        messagebox.showinfo("Grabando", f"Grabando palabra {palabra_n + 1}...")
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float64')
        sd.wait()
        audio = audio.flatten()
        audios.append(audio)
        freq, fft = calcular_fft(audio)
        fft_data.append(fft)

        mostrar_grafico_tiempo(audio, f"Palabra {palabra_n + 1} - Tiempo")
        mostrar_grafico_fft(freq, fft, f"Palabra {palabra_n + 1} - FFT")

        messagebox.showinfo("Guardado", f"Palabra {palabra_n + 1} guardada.")
    elif palabra_n == 3:
        messagebox.showinfo("Grabando", "Grabando palabra 4 (a comparar)...")
        audio4 = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float64')
        sd.wait()
        audio4 = audio4.flatten()

        freq4, fft4 = calcular_fft(audio4)
        mostrar_grafico_tiempo(audio4, "Palabra 4 - Tiempo")
        mostrar_grafico_fft(freq4, fft4, "Palabra 4 - FFT")

        comparar_fft(fft4)

# Calcular FFT
def calcular_fft(audio):
    N = len(audio)
    freq = np.fft.fftfreq(N, d=1/FS)
    fft = np.abs(np.fft.fft(audio))
    return freq[:N // 2], fft[:N // 2]

# Mostrar señal en el tiempo
def mostrar_grafico_tiempo(audio, title):
    fig = Figure(figsize=(4, 2), dpi=100)
    plot = fig.add_subplot(111)
    tiempo = np.linspace(0, DURATION, len(audio))
    plot.plot(tiempo, audio)
    plot.set_title(title)
    plot.set_xlabel("Tiempo (s)")
    plot.set_ylabel("Amplitud")
    canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Mostrar FFT (frecuencia vs amplitud)
def mostrar_grafico_fft(freq, fft, title):
    fig = Figure(figsize=(4, 2), dpi=100)
    plot = fig.add_subplot(111)
    plot.plot(freq, fft)
    plot.set_title(title)
    plot.set_xlabel("Frecuencia (Hz)")
    plot.set_ylabel("Amplitud")
    canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Comparar usando las FFTs (similitud de coseno)
def comparar_fft(fft4):
    similarities = []
    for i, fft_ref in enumerate(fft_data):
        # Normalización para evitar diferencias por volumen
        fft4_norm = fft4 / np.linalg.norm(fft4)
        fft_ref_norm = fft_ref / np.linalg.norm(fft_ref)
        sim = np.dot(fft4_norm, fft_ref_norm)
        similarities.append(sim)

    max_sim = max(similarities)
    if max_sim > 0.9:
        index = similarities.index(max_sim)
        messagebox.showinfo("Resultado", f"La palabra 4 es SIMILAR a la palabra {index + 1} (similitud = {max_sim:.2f})")
    else:
        messagebox.showinfo("Resultado", f"La palabra 4 es DIFERENTE (máxima similitud = {max_sim:.2f})")

# Interfaz
root = tk.Tk()
root.title("Reconocimiento de Palabras con FFT")

frame_botones = tk.Frame(root)
frame_botones.pack(pady=10)

frame_graficos = tk.Frame(root)
frame_graficos.pack()

# Botones
for i in range(4):
    btn = tk.Button(frame_botones, text=f"Grabar palabra {i + 1}", command=lambda i=i: grabar_y_guardar(i))
    btn.grid(row=0, column=i, padx=5)

root.mainloop()
