import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools


def n_tau_infty(V):
    """
    YA ESTA DESARROLLADA
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
    YA ESTA DESARROLLADA
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
##
'''
EN DESARROLLO
'''
def func_G(time, timespk):
    #deltat = 10 ** (-3)  # seg
    factor_temp = 12  # factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
    tausyn = 2.728 * 10**(-3)  # seg
    resultado = []
    for j in timespk.keys():
        resultado.append(sum((time - timespk[j])/(tausyn ** 2) * np.exp(- (time - timespk[j]) / tausyn) *
                             np.heaviside(time - timespk[j], 1)))
    return np.asarray(resultado)
"""
Vamos a proponer la red con una matriz de conexión, y un dict de actividad.
Lo que sí sabemos es que la neurona va a recibir un diccionario con los tiempos en que se dieron los spikes.
"""

def selector_spk(time, timespk, neurons = None):
    """
    Selecciona los spikes que van a entrar en consideración y genera una matriz
    :param time: tiempo actual
    :param timespk: dict de activad
    :param neurons: (optativo) vector con neuronas de interes
    :return:
    """

    deltat = 10 ** (-3)  # seg
    factor_temp = 12  # factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
    tausyn = 2.728 * 10**(-3)  # seg
    auxdic ={}
    if neurons==None:
        for j in timespk.keys():
            auxdic[j] = timespk[j][(abs(time - timespk[j]) < (factor_temp*tausyn)) & ((time - timespk[j]) > 0)]
    else:
        for j in neurons:
            auxdic[j] = timespk[j][(abs(time - timespk[j]) < (factor_temp * tausyn)) & ((time - timespk[j]) > 0)]
    return auxdic

def corr_syn(V, tiempo, timespk_dic = None, conductacia = None, neuronas = None):
    """
    corriente sinaptica. Es el aporte resultante de los potenciales de acción presinapticos.
    La conductancia va a estar asociada al aprendizaje.
    :param V:
    :param tiempo:
    :param timespk_dic:
    :param conductacia: por lo pronto es un vector
    :param neuronas:
    :return:
    """
    Ve = 0  # [mV] potencial reverso excitatorio
    conductacia_ref = 0.05*10**(-7)  # [S]
    if timespk_dic == None:
        return (-V+Ve)
    else:
        aux_spk_time = selector_spk(tiempo, timespk_dic, neuronas)  # spikes en la ventana de tiempo de proc
        aporte_pot = func_G(tiempo, aux_spk_time)
        return (-1) * (V - Ve) * aporte_pot.dot(conductacia)
#%%
##
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
cantidadspk = 20
spikes_time={}
for k in range(5):
    timespk, spikes = spike_train_generator(longsim, deltat, tinit, int(cantidadspk * np.random.rand()), freq, returnruster=True)
    spikes_time[k] = timespk

conductancias = np.asarray([1, 1, 1, 1, 1])*10 ** (-7) # una conductancia normal es de 0.05uS

#%%
##
vec_time = np.arange(0,1,deltat)
bb = []
for i in vec_time:
    aux = selector_spk(i, spikes_time)
    a = func_G(i, aux)
    bb.append(corr_syn(V, i, spikes_time, conductancias))

#%%
##

