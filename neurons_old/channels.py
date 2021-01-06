

import numpy as np
import torch


class Neurona():

    def __init__(self):
        # Parametros correspondientes al modelo integrate and fire
        self.gM = 0.0203 * 10 ** (-6)  # Conductancia muscarínica [S] (uS = 1/MOhm)
        self.Vk = -90  # Potencial de reversion del K (mV)
        self.C = 0.29 * 10 ** (-9)  # Capacitancia de membrana (F)
        self.gl = 0.029 * 10 ** (-6)  # Conductancia de leakage [S] (uS = 1/MOhm)
        self.Vl = -70  # Potencial de reversion de leakage (mV)
        self.VR = -60  # Potencial de reset (mV)

        # Umbral de simplificación del modelo, durante la generación del spike.
        self.Vswitch = -30  # Si self.V > Vswitch, el modelo se simplifica y posee una solución explícita.

        # Parametros correspondientes al termino exponencial
        # estos parametros tambien generan el spike en su forma
        self.Vt = -46  # umbral de spyke (mV)
        self.DVt = 3.6  # factor de pendiente (mV)

        # Parámetros de las corrientes sinapticas
        self.gsyn = 0.05 * 10 ** (-6)  # conductancia sinaptica excitatoria (S)
        self.conductacia_ref = 0.05 * 10 ** (-7)  # [S]
        self.tausyn = 2.728 * 10 ** (-3)  # constante temporal de la sinapsis excitatoria (seg)
        self.tau_nmda = 0.500 * 10 ** (-3)  # constante temporal para los canales NMDA

        # Corrientes
        self.Iext = 0  # una corriente externa.
        self.Im = 0  # corriente muscarinica.
        self.Iion = 0  # corriente ionica.
        self.Ispk = 0  # corriente sinaptica.
        self.INMDA = 0  # corriente NMDA

        self.V = self.Vl
        self.n = 0  # cantidad media de canales excitatorios abiertos.
        self.t = 0  # tiempo

        self.deltat = 0.001  # paso de la simulación [seg]

        self.neuron_idx = None  # indice identificador de la neurona.

        # ENTRADA DE DATOS A LA NEURONA.
        self.tspikes = np.asarray([])  # tiempos de spikes presinapticos
        self.tspikesm = np.asarray([])  # tiempos de spikes presinapticos muscarínicos
        self.tspikesd = np.asarray([])  # tiempos de spikes presinapticos dopaminérgicos

        # Conductacia sináptica
        self.conductaciasyn = np.asarray([])  # son los pesos sinápticos.

    