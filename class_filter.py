import scipy.signal as sp
import matplotlib.pyplot as plt 
import numpy as np

class FilterFIR():

    def __init__(self) -> None:
        pass

    def band_passs(self,M,fc1,fc2,tb,fs,plot=True):
        self.M = M
        self.fc1 = fc1
        self.fc2 = fc2
        self.tb=tb
        edges = [0, fc1 - tb, fc1, fc2,fc2 + tb, fs/2]
        
        self.h = sp.remez(self.M, edges, [0, 1,0], Hz=fs)
        ww, hw = sp.freqz(self.h, [1], worN=2000)

        if(plot==True):
            self.plot_response(fs,ww,hw," d")
        
    def plot_response(self,fs, w, h, title):
        "Utility function to plot response functions"
        fig = plt.figure()
        plt.subplot(2,1,1)

        plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
        #ax.plot(0.5*fs*w/np.pi, (np.abs(h)))
        plt.ylim(-40, 3)
        plt.xlim(0, 3000)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title(title)

        plt.subplot(2,1,2)
        h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
        plt.plot(0.5*fs*w/np.pi, h_Phase)
        #ax.plot(0.5*fs*w/np.pi, (np.abs(h)))
        plt.ylim(-40, 3)
        plt.xlim(0, 3000)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.title(title)