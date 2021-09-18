import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

#%%

def generador(tiempo,  canales, fs=1e3, freq = None, phi=None):
    '''
    tiempo es el ancho de la ventana de tiempo
    canales son cuantos canales hay que genera
    fs freuencia de muestreo
    phi la fase
    freq es la frecuencia de la envolvente
    '''
    if freq is None:
        freq = 2/tiempo  # 4 tramos de actividad.

    if phi is None:
        phi = torch.rand(canales,1) * 2 * np.pi  # fase de la actividad enganchada.
    Tsub = 10
    rango = torch.arange(0,int(tiempo*fs))
    rango = rango[::Tsub]

    out = torch.zeros((canales,int(tiempo*fs)))
    for i in range(canales):
        ss = torch.sin(2 * np.pi * freq/fs * rango + phi[i]) ** 2
        rr = torch.rand(ss.size())
        tt = rr < ss
        tt = tt * torch.randint(0,2, tt.size())
        out[i, ::Tsub] = tt
    return out

def plot_ruster(matx, minx=None, maxx=None):
    range = torch.arange(0,matx.size()[1])
    eventos = []
    fig = plt.figure( figsize=(12,8))
    for i in torch.arange(0,int(matx.size()[0])):
        eventos.append(range[matx[i,:]==1])
    colors2 = 'black'
    lineoffsets2 = 1
    linelengths2 = .1
    plt.eventplot(eventos, colors=colors2, lineoffsets=lineoffsets2, linelengths=linelengths2)




#%%

# Entrada de datos.
# Se va a generar una entrada de datos de 10 segundos a fs = 1kHz donde la mayor cantidad
# de canales va a tener actividad aleatoria, salvo dos que modularan la actividad temporal.

fs = 1e3
time_window = 10 # segundos
channel_in = 10  #  cantidad de canales
mtx = torch.rand(channel_in, int(time_window*fs))
channels_in = torch.zeros(mtx.size())
channels_in[mtx > 0.99] = 1

out = generador(tiempo=time_window, canales=3, fs=fs)

channels_in[3,:] = out[0,:]
channels_in[5,:] = out[1,:]
channels_in[7,:] = out[2,:]

del(mtx, out)

#%%

