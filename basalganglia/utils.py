import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

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
    plt.show()


#%%

