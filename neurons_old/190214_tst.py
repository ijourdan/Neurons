import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools
#%%


def n_tau_infty(V):
    """
    def n_tau_infty(self):
    Define n_infinito y Tau_n para el calculo de la cantidad promedio de canales muscarínicos abiertos.
    :return: (n_infty , tau)
    """

    if V == -30:
        alpha = 0.0001*9
        beta = 0.0001*9
    else:
        alpha = 0.0001 * (V + 30) / (1 - np.exp(-(V + 30) / 9))
        beta = -0.0001 * (V + 30) / (1 - np.exp((V + 30) / 9))

    n_infty = alpha / (alpha + beta)
    tau = 1 / (3 * (alpha + beta))

    return n_infty, tau



def spike_train_generator(tlong, deltat, tinit, cant_spk, freq, returnruster = False, trefractario = 0.006):
    """
    Genera un tren de spikes , identificados con 1, generados a partir de una distribución Poisson
    :param tlong: duracion de la señal generada en segundos
    :param deltat:
    :param tinit:
    :param cant_spk:
    :param freq: No debería ser mayor a 1/(deltat*10) 
    :return: returnruster = False: returns timespikes) where timespikes is a vector o times of spikes occurrence.
             returnruster = True: returns a tupla (timespikes,spikes), where spikes is a one channel spike ruster 
    """
    s = np.random.poisson(1/freq/deltat, cant_spk).astype(int) #en mseg
    spikes = np.zeros((np.int(tlong / deltat), 1))
    inicial = np.int(tinit / deltat)
    tspikes = []
    for i,j in zip(s,range(len(s))):
        while i <= (trefractario/deltat): # Tiempo refractario
            i = np.random.poisson(1/freq/deltat, 1).astype(int)[0]  # en mseg
        inicial = inicial + i
        if inicial < len(spikes):
            spikes[inicial] = 1
            tspikes.append(inicial)
    tspikes = np.asarray(tspikes)
    if returnruster:
        return  tspikes*deltat, spikes
    else:
        return  tspikes*deltat


#%%

#Datos para la simulacion
deltat = 0.001  # (seg) resolución temporal
longsim = 600  # (seg) tiempo de duración de la simulacion
V = -70

#Condiciones iniciales
n0, aux = n_tau_infty(V)
t = np.asarray([0, deltat])  # np.linspace(0, 2000, 5000)

# Genero un potencial de accion
freq = 10  # (Hz) Frecuencia media de disparo
tinit = 4  # (seg) tiempo donde comienza
cantidadspk = 1
tspikes, spikes = spike_train_generator(longsim, deltat, tinit, cantidadspk, freq, returnruster= True)


# Corremos la simulación

tacc = []
nacc = []
i = 0
while t[1] < longsim:  # en seg
    n = odeint(func_n, n0, t, args=(V, spikes[i],))
    n0 = n[1]
    t = t + deltat
    nacc.append(n0)
    tacc.append(t[0])
    i += 1


#%%
plt.plot(tacc, nacc)
#%%
def func_G(time, timespk):
    #deltat = 10 ** (-3)  # seg
    factor_temp = 12  # factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
    tausyn = 2.728 * 10**(-3)  # seg
    aux_timespk = timespk[(abs(time - timespk) < (factor_temp*tausyn)) & ((time - timespk) > 0)]
    return sum((time-aux_timespk)/(tausyn ** 2) * np.exp(- (time - aux_timespk) / tausyn) * np.heaviside(time - aux_timespk, 1))

#%%
#Datos para la simulacion
deltat = 0.001  # (seg) resolución temporal
longsim = 1  # (seg) tiempo de duración de la simulacion
V = -60

#Condiciones iniciales
n0, aux = n_tau_infty(V)
t = np.asarray([0, deltat])  # np.linspace(0, 2000, 5000)

# Genero un potencial de accion
freq = 10  # (Hz) Frecuencia media de disparo
tinit = 0.2#*np.random.rand()  # (seg) tiempo donde comienza
cantidadspk = 10
timespk, spikes = spike_train_generator(longsim, deltat, tinit, cantidadspk, freq, returnruster=True)

#%%
vec_time = np.arange(0,1,deltat)
bb = []
for i in vec_time:
    a = func_G(i, timespk)
    bb.append(a)
bb=np.asarray(bb)
#%%

def corr_syn(self,V, tiempo, pre_neurons = None, timespk_dic = None):
        # No sabemos bien como lo vamos a encarar
