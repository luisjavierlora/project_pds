import numpy as np
import matplotlib.pyplot as plt 
from scipy.io.wavfile import read,write
from PIL import Image
import scipy.signal as sp
from class_filter import FilterFIR
from class_instruments import Instrumento
import sounddevice as sd

class Audio():
    def __init__(self,path) -> None:
        self.path = path
        
    def read_audio(self):
        try:
            self.fs,self.signal = read(self.path) # leer audio
            self.stereo_to_mono() 
            self.normalizar_audio()

        except:
            print("Audio no fue cargado exitosamente")

    def stereo_to_mono(self):   
        if(len(self.signal.shape)>1): # tiene mas de dos canales
            canal = 0
            self.signal = self.signal[:,canal]
        
    def normalizar_audio(self):
        self.signal_norm = self.signal/(abs(self.signal).max())
    
    def save_grafica_audio(self):
        t = np.arange(0, len(self.signal_norm)/self.fs , 1/self.fs) #vector de tiempo
        """Recordar verificar que la carpeta cache exista"""
        plt.figure(figsize=(8,4)) # se define la figura
        plt.plot(t,self.signal_norm,"b",linewidth=2) # se genera el estilo de la figura      
        plt.grid()
        plt.title("Senal Audio")
        plt.xlabel("Time [s]",fontsize=12)
        plt.ylabel("Amplitud [a.u]",fontsize=12) 

        plt.savefig('cache/testplot.png') #se guarda la figura
        Image.open('cache/testplot.png').convert('RGB').save('cache/testplot.jpg')
       


class Frames():
    def __init__(self,audio_class,time_window,step_time = 0.5e-3) -> None:
        self.signal = audio_class.signal_norm
        self.fs = audio_class.fs
        self.time_window = time_window
        self.step_time = step_time
        self.size = int(self.fs*time_window)

    def extraer_ventanas(self):
        
        #Comprobando que sea senial mono (1 canal)
        assert(self.signal.ndim == 1)
        
        #Tamano de paso
        #step=int(0.050*fs)
        self.step = int(self.step_time*self.fs)

        n_seg = int((len(self.signal) - self.size) / self.step)
        print("numero de ventanas: ",n_seg)
    
        # extraer segmentos
        self.windows=[]
        for i in range(n_seg):
            self.windows.append(self.signal[i * self.step : i * self.step + self.size])

        # stack (cada fila es una ventana)
        return np.vstack(self.windows)

    def get_windows(self):
        return np.vstack(self.windows)



def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]



audio = Audio("audios/corazon_encantado.wav")
audio.read_audio()
audio.save_grafica_audio()

# Filtro para eliminar el ruido
noise_filter=FilterFIR()
noise_filter.band_passs(500,250,1990,50,audio.fs)
h_noise_filter=noise_filter.h

y_filter = sp.lfilter(h_noise_filter, [1], audio.signal_norm)   
audio.signal_norm = y_filter

# Enventanado 
"""La ventanaa debe tomar al menos dos periodos de la señal para que la correlación funcione bien"""

f0_min = 262 # f0 minima de las notas del instrumento 
timeW = 6*(1/f0_min)  # 14 periodos

frames = Frames(audio,timeW,timeW*0.7)
list_windows = frames.extraer_ventanas() 

# Autocorrelación para hallar la frecuencia fundamental

fundamentals_freq = list()
for v in list_windows:

    auto_corr = autocorr(v)
    fft_autocorr=np.fft.fft(auto_corr*np.hamming(len(auto_corr)))
    freq = np.fft.fftfreq(auto_corr.size, d=1/audio.fs)
    mag = abs(fft_autocorr)
    index_maximo= np.argmax(mag)
    maximo = mag[index_maximo]

    if(maximo > 50):
        fundamental_freq = (abs(freq[index_maximo]))
        print(maximo,fundamental_freq)
        #fundamentals_freq.append(fundamental_freq)
        fundamentals_freq = fundamentals_freq + frames.step*[fundamental_freq]
    else:
        #print(0)
        #fundamentals_freq.append(0)
        fundamentals_freq = fundamentals_freq + frames.step*[0]


#time = len(list_windows)*frames.step*(1/frames.fs)
#t_fundamentals = np.arange(0,)
# se grafica junto la señal de audio
audio2 = audio.signal_norm[0:len(fundamentals_freq)]
t2 = np.arange(0, len(audio2)/audio.fs , 1/audio.fs)

piano = Instrumento()
piano.agregar_notas("notas/guitarra_acustica")


filtro_piano=FilterFIR()

sound_final = list()

for f0 in fundamentals_freq:
    if(f0!=0):
        if(abs(piano.nota_anterior-piano.nota_actual) >20 ):
            filtro_piano.band_passs(500,f0*0.5-15,f0*0.5+15,50,44100,plot=False)

        nota=piano.encontrar_esta_nota(f0*0.5,filtro_piano)
        frecuencia_armonicos,phase,armonicos_amplitud = nota.get_my_armonics(f0,8)
        piano.actualizar_tiempo()
        sound_piano = piano.tocar_nota(f0*0.5,armonicos_amplitud,phase)
        sound_final.append(sound_piano)
    else:
        sound_final.append(0)
        piano.t=0

sound_final =np.array(sound_final)
#print("max",max(abs(sound_final)))
#sound_final =sound_final/max(abs(sound_final))
sd.play( sound_final,44100)
plt.show()



"""
plt.figure()
plt.plot(t2,fundamentals_freq)
plt.plot(t2,audio2*100)
plt.plot(t2,np.array(sound_final)*100)
plt.grid()
plt.show()
"""



print(len(audio.signal_norm),len(fundamentals_freq))
#print("Time",time)