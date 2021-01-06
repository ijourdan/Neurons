import numpy as np

import matplotlib.pyplot as plt
import itertools
import spiny


##
#%%
#Datos para la simulacion

deltat = 0.001  # (seg) resolución temporal
longsim = 0.3  # (seg) tiempo de duración de la simulacion

#Condiciones iniciales

#t = np.asarray([0, deltat])  # np.linspace(0, 2000, 5000)

# Genero un potencial de accion
freq = 100  # (Hz) Frecuencia media de disparo
tinit = 0.2 #*np.random.rand()  # (seg) tiempo donde comienza
cantidadspk = 2
spikes_time={}
for k in range(1):
    timespk, spikes = spiny.EIaF.spike_train_generator(longsim, deltat, tinit, int(cantidadspk * np.random.rand()), freq, returnruster=True)
    spikes_time[k] = np.asarray(timespk)

conductancias = np.asarray([1])*0.05*10 ** (-6) # una conductancia normal es de 0.05uS

timespkm = spiny.EIaF.spike_train_generator(longsim, deltat, tinit, int(cantidadspk * np.random.rand()), freq, returnruster=True)
timespkm = np.asarray([-10])#timespkm[0]
#%%
##
neurona = spiny.EIaF()
neurona.deltat = deltat
neurona.tspikes = spikes_time
neurona.tspikesm = timespkm
neurona.conductaciasyn = conductancias

vec_time = np.arange(0,longsim,deltat)

#%%
##

aux_pot = []
aux_current = []
aporte_pot = []
y2 =[]
y3 =[]

for neurona.t in vec_time:

    y2.append(neurona.corr_syn(neurona.tspikes)) # no parece andar mal, salvo alguna unidad+
    y3.append(neurona.corr_syn_NMDA(neurona.tspikes, neurona.conductaciasyn)) # no parece andar mal, salvo alguna unidad

    aux_spk_time = neurona.selector_spk(neurona.t, neurona.tspikes)  # spikes en la ventana de tiempo de proc
    if len(aux_spk_time[0]) > 0:
        print(aux_spk_time)
    aporte_pot.append(neurona.func_G(neurona.t, aux_spk_time))
    aux_current.append(neurona.func_GNMDA(neurona.t, aux_spk_time))
    #aux_pot.append()




#%%
##

plt.plot(aux_current)
plt.plot(aporte_pot)
##

plt.plot(y2)
plt.plot(y3)
##

