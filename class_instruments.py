
from typing import FrozenSet
from os import listdir
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax 
from scipy.io.wavfile import read,write
import numpy as np
from class_filter import FilterFIR
import scipy.signal as sp
import sounddevice as sd

def ls(ruta = '.'):
    return listdir(ruta)

class Nota:
    def __init__(self) -> None:
        self.armonics = None # lista de los 5 primeros armonicos
        self.fs = None #frecuencia fundamental
        self.signal_note_fft = list()
        self.peaks_index = list()
    

    def read_nota(self,path):
        try:
            self.path = path
            self.fs,self.signal_nota = read(path) # leer audio
            self.stereo_to_mono() 
            self.normalizar_audio()

        except:
            print("Audio no fue cargado exitosamente") 

    def stereo_to_mono(self):   
        if(len(self.signal_nota.shape)>1): # tiene mas de dos canales
            canal = 0
            self.signal_nota = self.signal_nota[:,canal]
        
    def normalizar_audio(self):
        self.signal_note_norm = self.signal_nota/(abs(self.signal_nota).max())
    
    def get_my_armonics(self,f0,npeaks):

        if(len(self.peaks_index)==0):
            signal_note_fft = self.my_fft()
            self.magnitud = np.abs(signal_note_fft)
            #self.fase = np.arctan2(np.imag(signal_note_fft),np.real(signal_note_fft))
            self.fase = np.unwrap(np.arctan2(np.imag(signal_note_fft),np.real(signal_note_fft)))
            freq_signal_audio = np.fft.fftfreq(self.magnitud.size)*self.fs



            self.peaks_index, _ = sp.find_peaks(self.magnitud[0:len(self.magnitud)//2],height=5, distance = 100)
            self.peaks_freq=freq_signal_audio[self.peaks_index]

            #encontrar maximo
            media_fft=self.magnitud[:len(self.magnitud)//2]
            index_max=np.argmax(media_fft)
            f0=freq_signal_audio[index_max]

            peaks_index = []
            cont=1
            peaks_freq = []

            for iter,value in enumerate(self.peaks_freq):
                if value > f0-5 :
                    peaks_index.append(self.peaks_index[iter])
                    peaks_freq.append(value)

            self.peaks_index = peaks_index
            self.peaks_freq = peaks_freq
            if(len(self.peaks_index)>  npeaks):
                self.peaks_index=self.peaks_index[0:npeaks]
                self.peaks_freq=self.peaks_freq[0:npeaks]

            print(self.peaks_freq , self.path)

        #fft=np.fft.fft(self.signal_note_norm)
        #m= abs(fft)
        return self.peaks_index,self.fase[self.peaks_index], self.magnitud[self.peaks_index]/(self.magnitud[self.peaks_index][0])

    def my_fft(self):

        if(len(self.signal_note_fft)==0):
            ventana = self.signal_note_norm*np.hamming(len(self.signal_note_norm))
            self.signal_note_fft=np.fft.fft(ventana)
            
        
        return self.signal_note_fft
            

class Instrumento:
    def __init__(self) -> None:
        self.nota_anterior = 0
        self.nota_actual = 21
        self.all_my_notes = list()

    def agregar_notas(self,path):
        self.notas_wav = ls(path)
        print(self.notas_wav)
        for i in self.notas_wav:
            new_nota=Nota()
            new_nota.read_nota(path + "/"+i)
            self.all_my_notes.append(new_nota)

    def actualizar_tiempo(self):
        
        if(abs(self.nota_anterior-self.nota_actual) >20 ):
            self.t = 0
        else:
            self.t= self.t + 1/44100

    def encontrar_esta_nota(self,f0,filter_FIR):
        if(abs(self.nota_anterior-self.nota_actual) >20 ):
            magnitudes=[]
            for nota in self.all_my_notes:
                ventana= nota.signal_note_norm *np.hamming(len(nota.signal_note_norm))
                
                y_filter = sp.lfilter(filter_FIR.h, [1], ventana)   
                
                magnitud_y=np.abs(np.fft.fft(y_filter))
                magnitudes.append(max(magnitud_y))

            self.index_max=np.argmax(magnitudes)
            return self.all_my_notes[self.index_max]

        else:
            return self.all_my_notes[self.index_max]

    def tocar_nota(self,frec_funda,amp_harm,fase):
        self.nota_anterior = self.nota_actual
        self.nota_actual = frec_funda

        """        wave_audio = amp_harm[0]*np.cos(2*np.pi*frec_funda*self.t - fase[0])

        for i in range(1,len(amp_harm)):
            if(i%2 !=0):
                wave_audio =wave_audio+ amp_harm[i]*np.sin((i+1)*2*np.pi*frec_funda*self.t)
            else:
                wave_audio =wave_audio+ amp_harm[i]*np.cos((i+1)*2*np.pi*frec_funda*self.t)"""
        
        """
        for k in range(0,len(amp_harm)):
             wave_audio = amp_harm[k]*np.exp(1j*2*np.pi*(k+1)*frec_funda*self.t)
        """
        wave_audio=0
        for k in range(0,len(amp_harm)):
            wave_audio =wave_audio+ amp_harm[k]*np.cos(2*np.pi*(k+1)*frec_funda*self.t - fase[k])

        wave_audio = wave_audio*np.exp(-2.30*self.t/0.5)  #dura 4 tiempos
        #wave_audio =wave_audio+ amp_harm[3]*np.sin(8*np.pi*frec_funda*t)
        #wave_audio =wave_audio+ amp_harm[4]*np.cos(10*np.pi*frec_funda*t)
        #wave_audio =wave_audio+ amp_harm[5]*np.sin(12*np.pi*frec_funda*t)
        #wave_audio =wave_audio+ amp_harm[6]*np.cos(14*np.pi*frec_funda*t)

        return wave_audio

"""
piano = Instrumento()
piano.agregar_notas("notas/piano")

# Filtro para notas del piano
f0=523
filtro_piano=FilterFIR()
filtro_piano.band_passs(500,f0-15,f0+15,50,44100)

sound_final = list()

for i in range(0,44100):
    nota=piano.encontrar_esta_nota(f0,filtro_piano)
    if(i==0):
        frecuencia_armonicos,armonicos_amplitud = nota.get_my_armonics(f0,6)
    sound_piano = piano.tocar_nota(f0,armonicos_amplitud)
    sound_final.append(sound_piano)


sd.play(sound_final,44100)
plt.show()
"""
#print(sound_piano)
#print(nota.path)
#print(frecuencia_armonicos,armonicos_amplitud)
#plt.plot(sound_final)

"""
class Piano:
    def __init__(self) -> None:
        self.instrumento = Instrumento()
        self.notas_path = "notas/piano" 
        
    def cargar_notas(self):
        self.instrumento.agregar_notas(self.notas_path)

"""


   
