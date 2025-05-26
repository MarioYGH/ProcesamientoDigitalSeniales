# Este código adquiere el 3 palabras, por audio, saca su frecuencia y la guarda y después se debe ingresar una cuarta palabra para comparar si es 
# igual a una de las ya antes guardadas, en la funcion def mostrar resultado, se puede ajustar el umbral para un mejor resultado

import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Tuple

# Configuración global
DURATION = 1.5  
FS = 44100
CHANNELS = 1
DTYPE = 'float64'

class AudioProcessor:
    def __init__(self):
        self.audios: List[np.ndarray] = []       # Señales originales
        self.fft_data: List[Tuple[np.ndarray, np.ndarray]] = []  # (freq, fft)
        
    def grabar_audio(self) -> np.ndarray:
        """Graba audio con la configuración global"""
        audio = sd.rec(int(DURATION * FS), samplerate=FS, 
                       channels=CHANNELS, dtype=DTYPE)
        sd.wait()
        return audio.flatten()
    
    @staticmethod
    def calcular_fft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula la FFT de una señal de audio"""
        N = len(audio)
        freq = np.fft.fftfreq(N, d=1/FS)
        fft = np.abs(np.fft.fft(audio))
        return freq[:N//2], fft[:N//2]
    
    def comparar_fft(self, fft4: np.ndarray) -> Tuple[int, float]:
        """Compara una FFT con las almacenadas usando similitud de coseno"""
        similarities = []
        for i, (_, fft_ref) in enumerate(self.fft_data):
            # Normalización para evitar diferencias por volumen
            fft4_norm = fft4 / np.linalg.norm(fft4)
            fft_ref_norm = fft_ref / np.linalg.norm(fft_ref)
            sim = np.dot(fft4_norm, fft_ref_norm)
            similarities.append(sim)
        
        max_sim = max(similarities) if similarities else 0
        index = similarities.index(max_sim) if similarities else -1
        return index, max_sim
    
class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Palabras con FFT")
        self.processor = AudioProcessor()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura los elementos de la interfaz gráfica"""
        self.frame_botones = tk.Frame(self.root)
        self.frame_botones.pack(pady=10)
        
        self.frame_graficos = tk.Frame(self.root)
        self.frame_graficos.pack()
        
        # Botones para grabar
        for i in range(4):
            btn = tk.Button(
                self.frame_botones, 
                text=f"Grabar palabra {i + 1}", 
                command=lambda i=i: self.grabar_y_procesar(i),
                width=15
            )
            btn.grid(row=0, column=i, padx=5)
            
        # Botón para limpiar
        btn_clear = tk.Button(
            self.frame_botones,
            text="Limpiar todo",
            command=self.limpiar_datos,
            width=15
        )
        btn_clear.grid(row=1, column=0, columnspan=4, pady=5)
        
    def grabar_y_procesar(self, palabra_n: int):
        """Maneja el proceso de grabación y procesamiento"""
        try:
            if palabra_n < 3:
                self.mostrar_mensaje("Grabando", f"Preparado para grabar palabra {palabra_n + 1}...")
                audio = self.processor.grabar_audio()
                self.processor.audios.append(audio)
                
                freq, fft = self.processor.calcular_fft(audio)
                self.processor.fft_data.append((freq, fft))
                
                self.mostrar_grafico_tiempo(audio, f"Palabra {palabra_n + 1} - Tiempo")
                self.mostrar_grafico_fft(freq, fft, f"Palabra {palabra_n + 1} - FFT")
                
                self.mostrar_mensaje("Éxito", f"Palabra {palabra_n + 1} guardada.")
                
            elif palabra_n == 3:
                if len(self.processor.fft_data) < 3:
                    self.mostrar_mensaje("Error", "Primero grabe las 3 palabras de referencia")
                    return
                    
                self.mostrar_mensaje("Grabando", "Preparado para grabar palabra 4 (a comparar)...")
                audio4 = self.processor.grabar_audio()
                
                freq4, fft4 = self.processor.calcular_fft(audio4)
                self.mostrar_grafico_tiempo(audio4, "Palabra 4 - Tiempo")
                self.mostrar_grafico_fft(freq4, fft4, "Palabra 4 - FFT")
                
                index, sim = self.processor.comparar_fft(fft4)
                self.mostrar_resultado(index, sim)
                
        except Exception as e:
            self.mostrar_mensaje("Error", f"Ocurrió un error: {str(e)}")
    
    def mostrar_grafico_tiempo(self, audio: np.ndarray, title: str):
        """Muestra la gráfica en el dominio del tiempo"""
        self.limpiar_graficos()
        fig = Figure(figsize=(6, 3), dpi=100)
        plot = fig.add_subplot(111)
        tiempo = np.linspace(0, DURATION, len(audio))
        plot.plot(tiempo, audio, color='blue')
        plot.set_title(title)
        plot.set_xlabel("Tiempo (s)")
        plot.set_ylabel("Amplitud")
        canvas = FigureCanvasTkAgg(fig, master=self.frame_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def mostrar_grafico_fft(self, freq: np.ndarray, fft: np.ndarray, title: str):
        """Muestra la gráfica en el dominio de la frecuencia"""
        fig = Figure(figsize=(6, 3), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(freq, fft, color='green')
        plot.set_title(title)
        plot.set_xlabel("Frecuencia (Hz)")
        plot.set_ylabel("Amplitud")
        plot.set_xlim(0, 5000)  # Limitamos a frecuencias relevantes del habla
        canvas = FigureCanvasTkAgg(fig, master=self.frame_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def mostrar_resultado(self, index: int, similitud: float):
        """Muestra el resultado de la comparación"""
        if similitud > 0.85:  # Umbral ajustable
            msg = f"La palabra 4 es SIMILAR a la palabra {index + 1} (similitud = {similitud:.2f})"
        else:
            msg = f"La palabra 4 es DIFERENTE (máxima similitud = {similitud:.2f})"
        self.mostrar_mensaje("Resultado", msg)
    
    def limpiar_datos(self):
        """Limpia todos los datos y gráficos"""
        self.processor.audios.clear()
        self.processor.fft_data.clear()
        self.limpiar_graficos()
        self.mostrar_mensaje("Información", "Datos y gráficos limpiados")
    
    def limpiar_graficos(self):
        """Elimina todos los gráficos mostrados"""
        for widget in self.frame_graficos.winfo_children():
            widget.destroy()
    
    @staticmethod
    def mostrar_mensaje(titulo: str, mensaje: str):
        """Muestra un mensaje en ventana emergente"""
        messagebox.showinfo(titulo, mensaje)
        
def main():
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
