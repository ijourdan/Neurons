import numpy as np


# import scipy as sp


# import matplotlib.pyplot as plt


class EIaF(object):

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
        self.tau_nmda = 0.500 * 10 **(-3)  # constante temporal para los canales NMDA

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

    @staticmethod
    def spike_train_generator(tlong, deltat, tinit, cant_spk, freq, returnruster=False, trefractario=0.006):
        """
        Genera un tren de spikes , identificados con 1, generados a partir de una distribución Poisson
        :param tlong: duracion de la señal generada en segundos
        :param deltat:
        :param tinit:
        :param cant_spk:
        :param freq:
        :param freq: No debería ser mayor a 1/(deltat*10)
        :param returnruster:
        :param trefractario: Tiempo mínimo entre spikes por cuestiones refractarioas. [seg]. Default = 0.006 seg
        :return: returntime = False: returns spikes, a one channel spike ruster
                 returntime = True: returns a tupla (spikes, timespikes) where timespikes is a vector o times
                 of spikes occurrence.
        """
        s = np.random.poisson(1 / freq / deltat, cant_spk).astype(int)  # en mseg
        spikes = np.zeros((np.int(tlong / deltat), 1))
        inicial = np.int(tinit / deltat)
        tspikes = []
        for i, j in zip(s, range(len(s))):
            while i <= (trefractario / deltat):  # Tiempo refractario
                i = np.random.poisson(1 / freq / deltat, 1).astype(int)[0]  # en mseg
            inicial = inicial + i
            if inicial < len(spikes):
                spikes[inicial] = 1
                tspikes.append(inicial)
        tspikes = np.asarray(tspikes)
        if returnruster:
            return tspikes * deltat, spikes
        else:
            return tspikes * deltat

    def info(self, infomin=False):
        if not (infomin):
            print('Parámetros de estado de neurona:')
            print('')
            print('Idx neurona: ' + str(self.neuron_idx))
            print('Potencial de membrana [mV]: ' + str(self.V))
            print('Tiempo [seg]' + str(self.t))
            print('Paso de la simulación [seg]: ' + str(self.deltat))
        else:
            print('Parámetros de estado de neurona:')
            print('')
            print('Idx neurona: ' + str(self.neuron_idx))
            print('Potencial de membrana [mV]: ' + str(self.V))
            print('Tiempo [seg]' + str(self.t))
            print('Paso de la simulación [seg]: ' + str(self.deltat))
            print('')
            print('=========================================================')
            print('')
            print('Parámetros de membrana de la neurona:')
            print('')
            print('Potencial de reversion del K (mV): ' + str(self.Vk))
            print('Capacitancia de membrana (nF): ' + str(self.C))
            print('Potencial de reset (mV): ' + str(self.VR))
            print('Umbral de spyke (mV): ' + str(self.Vt))
            print('')
            print('=========================================================')
            print('')
            print('Parametros correspondientes a las corrientes:')
            print('')
            print('Corriente Muscaríninca (Im)[mA]: ' + str(self.Im))
            print('Conductancia muscarínica (gM) [S](uS = 1/MOhm): ' + str(self.gM))
            print('Cantidad media de canales excitatorios muscarínicaos activos (n): ' + str(self.n))
            print('')
            print('Corriente ionica(incluye leak) (Iion) [mA]: ' + str(self.Iion))
            print('Conductancia de leakage (gl) [S] (uS = 1/MOhm): ' + str(self.gl))
            print('Potencial de reversion de leakage (mV): ' + str(self.Vl))
            print('Factor de pendiente (mV): ' + str(self.DVt))
            print('')
            print('Corriente sinapticas Ispk [mA]: ' + str(self.Ispk))
            print('Conductancia sinaptica excitatoria (S): ' + str(self.gsyn))
            print('constante temporal de la sinapsis excitatoria (seg): ' + str(self.tausyn))
            print('')
            print('Corriente externa Iext [mA]: ' + str(self.Iext))
            print('')
            print('=========================================================')

    def integ_os_RK(self, funcion, y, x, h, args=None):
        """
        Itegra de forma one_step empleando Runge-Kutta de cuarto orden
        :param funcion: función a integrar. Se debe diseñar de manera tal que sus argumentos sean
        funcion(x,y,arg0,arg1, ...)

        la funcion integ_os_RK presupone que el argumento x es actualizado, y por lo tanto determina:

        y(x) = function(x,y(x-h),h, arg0,...)

        :param y: y(x-h) valor o condicion inicial para el paso de integración, correspondiente a y(x-h)
        :param x: argumento de la función y(x)
        :param h: variación o paso de integración sobre el argumento x
        :param args: (arg0,arg,1,arg2,) tupla con otros argumentos de funcion
        :return: y(x)
        """
        xn = x - h
        if args is None:
            k1 = h * funcion(xn, y)
            k2 = h * funcion(xn + h / 2, y + k1 / 2)
            k3 = h * funcion(xn + h / 2, y + k2 / 2)
            k4 = h * funcion(xn + h, y + k3)
            y_next = y + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        else:
            k1 = h * funcion(xn, y, *args)
            k2 = h * funcion(xn + h / 2, y + k1 / 2, *args)
            k3 = h * funcion(xn + h / 2, y + k2 / 2, *args)
            k4 = h * funcion(xn + h, y + k3, *args)
            y_next = y + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        return y_next

    def func_n(self, tspikes):

        """
        Determina n(t+dt).
        :param tspikes: vector de tiempo de realización de spikes.
        :return: n(t+dt)
        """

        def dfunc_n(n, t, V, spk=False):
            """
            Funcion derivada de la cantidad media de canales abiertos

            :param n:
            :param t:
            :param V:
            :param spk:
            :return: dn/dt
            """
            j = 0.014  # constante de salto

            def n_tau_infty(V):
                """
                def n_tau_infty(self):
                Define n_infinito y Tau_n para el calculo de la cantidad promedio de canales muscarínicos abiertos.
                :param V: Potencial de membrana (mV)
                :return:  (n_infty , tau)
                """

                if V == -30:
                    alpha = 0.0001 * 9
                    beta = 0.0001 * 9
                else:
                    alpha = 0.0001 * (V + 30) / (1 - np.exp(-(V + 30) / 9))
                    beta = -0.0001 * (V + 30) / (1 - np.exp((V + 30) / 9))

                n_infty = alpha / (alpha + beta)
                tau = 1 / (3 * (alpha + beta))

                return n_infty, tau

            n_infty, tau = n_tau_infty(V)
            return ((n_infty - n) / tau) + j * spk

        # vamos a determinar si hay un spike en el instante
        try:
            no_spike = (len(tspikes) == 0)
        except TypeError:
            no_spike = (tspikes == False)

        if no_spike:
            spikes = False
        else:
            aux = []
            for i in tspikes:
                aux.append((i < (self.t + self.deltat / 2)) & (
                        i > (self.t - self.deltat / 2)))  # detectamos si hay algun spike
            spikes = max(aux)  # si hay spikes, entonces se pone spikes en True,
        n = self.integ_os_RK(dfunc_n, self.n, self.t, self.deltat, args=(self.V, spikes,))
        return n

    def corr_Im(self, tspikes):
        self.n = self.func_n(tspikes)
        self.Im = self.gM * self.n * (self.V - self.Vk)
        return self.Im

    # corriente ionica (leakin)

    def corr_ion(self, V):
        """
        Función diferencial para las corrientes iónicas
        el la correinte iónica compuesta por Na, K y leaking
        :param V: Potencial de membrana
        :return: apote a dV/dt
        """
        self.Iion = self.gl * (self.Vl - V) 
        # Vamos a eliminar la parte exponencial#+ self.gl * self.DVt * np.exp((V - self.Vt) / self.DVt)
        return self.Iion

    # corriente sinaptica

    def func_G(self, tiempo, timespk):
        """
        Determina el proporcional de efecto del tren de potenciales de acción para el tiempo indicado.
        Considera una constante de crecimiento de tausyn = 2.728 * 10 ** (-3)  # seg y genera el aporte de spikes en
        una venta de hasta factor_temp * tausyn.
        :param tiempo: (float) tiempo (en segundos) donde se busca determinar el aporte de los potenciales de acción.
        :param timespk: (numpy.array)
        :return:
        """
        # factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
        resultado = []
        for j in timespk.keys():
            resultado.append(sum((tiempo - timespk[j]) / (self.tausyn) *
                                 np.exp(- (tiempo - timespk[j]) / self.tausyn) * np.heaviside(tiempo - timespk[j], 1)))
        return np.asarray(resultado)

    def func_GNMDA(self, tiempo, timespk):
        """
        Determina el proporcional de efecto del tren de potenciales de acción para el tiempo indicado.
        Considera una constante de crecimiento de tausyn = 2.728 * 10 ** (-3)  # seg y genera el aporte de spikes en
        una venta de hasta factor_temp * tausyn.
        :param tiempo: (float) tiempo (en segundos) donde se busca determinar el aporte de los potenciales de acción.
        :param timespk: (numpy.array)
        :return:
        """
        # factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
        resultado = []
        for j in timespk.keys():
            resultado.append(sum((tiempo - timespk[j]) / self.tau_nmda *
                                 np.exp(- (tiempo - timespk[j]) / self.tau_nmda) * np.heaviside(tiempo - timespk[j], 1)))
        return np.asarray(resultado)

    """
    Vamos a proponer la red con una matriz de conexión, y un dict de actividad.
    Lo que sí sabemos es que la neurona va a recibir un diccionario con los tiempos en que se dieron los spikes.
    """

    def selector_spk(self, time, timespk, neurons=None, factor_temp=12):
        """

        :param time:
        :param timespk:
        :param neurons:
        :param factor_temp: (=12 dflt) factor_temp*tausyn es la ventana donde considero que tiene efecto un spyke
        :return:
        """
        auxdic = {}
        if neurons == None:
            for j in timespk.keys():
                auxdic[j] = timespk[j][
                    (abs(time - timespk[j]) < (factor_temp * self.tausyn)) & ((time - timespk[j]) > 0)]
        else:
            for j in neurons:
                auxdic[j] = timespk[j][
                    (abs(time - timespk[j]) < (factor_temp * self.tausyn)) & ((time - timespk[j]) > 0)]
        return auxdic

    def corr_syn(self, timespk_dic=None,  neuronas=None):
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
        V = self.V
        tiempo = self.t
        Ve = 0  # [mV] potencial reverso excitatorio

        if type(timespk_dic) == 'NoneType':
            self.Ispk = (-V + Ve)
        else:
            aux_spk_time = self.selector_spk(tiempo, timespk_dic, neuronas)  # spikes en la ventana de tiempo de proc
            aporte_pot = self.func_G(tiempo, aux_spk_time)
            self.Ispk = (-1) * (V - Ve) * self.gsyn * aporte_pot

        return self.Ispk

    def corr_syn_NMDA(self, timespk_dic=None, conductancia=None, neuronas=None):
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
        V = self.V
        tiempo = self.t
        Ve = 0  # [mV] potencial reverso excitatorio

        if type(timespk_dic) == 'NoneType':
            self.INMDA = (-V + Ve)
        else:
            aux_spk_time = self.selector_spk(tiempo, timespk_dic, neuronas)  # spikes en la ventana de tiempo de proc
            aporte_pot = self.func_G(tiempo, aux_spk_time)
            self.INMDA = (-1) * (V - Ve) * aporte_pot.dot(conductancia)
        return self.INMDA

    def corr_i(self, V, t):
        aux = self.corr_ion(self.V)  + self.corr_syn_NMDA(timespk_dic=self.tspikes, conductancia=self.conductaciasyn) + self.corr_syn(timespk_dic=self.tspikes)# + self.Iext #- self.corr_Im(self.tspikesm)
        #aux = self.corr_syn(timespk_dic=self.tspikes, conductancia=self.conductaciasyn)
        return aux

    def der_pot_v(self,V,t):
        return 1/self.C * self.corr_i(V,t)

    def pot_v(self):
        v = self.integ_os_RK(self.der_pot_v, self.V, self.t, self.deltat)
        self.V = v
        return self.V
