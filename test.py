#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal
from utils import generador, plot_ruster

from units import InputUnits

from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

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

#channels_in[3,:] = out[0,:]
#channels_in[5,:] = out[1,:]
#channels_in[7,:] = out[2,:]

del(mtx, out)

#%%
# Se van a declarar dos unidades de entradas de datos.

input_units = InputUnits(n=10, traces=True)

#%%
longitud = channels_in.size()[1]
traces_out = torch.zeros((channel_in, longitud))
spikes_out =torch.zeros((channel_in, longitud))
for i in torch.arange(0,longitud):
    input_units.step(inputs=channels_in[:, i], mode='train', dt=1)
    traces_out[:, i] = input_units.get_traces()



#%%
#spk = pad_sequence(spikes_out)
#trc = pad_sequence(traces_out)

#%%




