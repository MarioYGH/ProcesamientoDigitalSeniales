import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Configuraciones globales
DURATION = 3  # segundos
FS = 44100    # frecuencia de muestreo

def grabar_audio():
    messagebox.showinfo("Grabando", f"Grabando {DURATION} segundos...")
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float64')
    sd.wait()
    messagebox.showinfo("Fin", "Grabación terminada.")
    calcular_dft(audio.flatten())

def calcular_dft(audio):
    N = len(audio)
    freq = np.fft.fftfreq(N, d=1/FS)
    spectrum = np.abs(np.fft.fft(audio))

    # Solo la mitad positiva
    half_N = N // 2
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:half_N], spectrum[:half_N])
    plt.title("Espectro de Frecuencia (DFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.show()

# Interfaz Tkinter
root = tk.Tk()
root.title("Grabador y Analizador de Audio")

label = tk.Label(root, text="Presiona el botón para grabar audio y ver su DFT")
label.pack(pady=20)

boton_grabar = tk.Button(root, text="Grabar Audio", command=grabar_audio)
boton_grabar.pack(pady=20)

root.mainloop()
