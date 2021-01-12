import torch

from abc import ABC, abstractmethod


class Unit(ABC):
    """
    Abstract base class for groups of neurons
    """

    def __int__(self, n, traces, trace_tc):
        super().__init__()

    @abstractmethod
    def step(self, inpts, mode, dt):
        pass


class InputUnits(Unit):
    def __int__(self, n, traces=False, trace_tc=5e-2):
        """
        Interface units for network input. This units are used to translate binary inputs to spikes.
        :param n: number of neurons
        :param traces: (True / False) if "True" initialize the unit traces for STDP learning
        :param trace_tc: Rate of decay of spike trace time constant
        :return: Nothing
        """
        super().__init__()
        self.n = n
        self.s = torch.zeros(n)  # spike vector.
        if traces:
            self.x = torch.zeros(n)
            self.trace_tc = trace_tc

    def step(self, inputs, mode, dt):
        """
        On each simulation step, sets the spikes of the population equal to the input.
        :param inputs: boolean or byte vector with the information
        :param mode: 'train' : if is 'train', updates the training traces.
        :param dt: time step
        :return: nothing
        """
        self.s = inputs
        if mode == 'train':
            self.x -= dt * self.trace_tc * self.x  # update spike traces
            self.x[self.s.byte()] = 1  # setting synaptic traces for a spike occurrence.

    def get_spikes(self):
        return self.s

    def get_traces(self):
        return self.x


class LIFUnits(Unit):
    def __int__(self, n, traces=False, trace_tc=5e-2, rest=-90.0, reset=-65.0, threshold=-60.0, refractory=5,
                voltage_decay=2.9e-2):
        """
        Leaky integrate and fire units. This units are used to translate binary inputs to spikes.
        The default configuration is for a striatal spiny neuron.
        :param n: number of neurons
        :param traces: (True / False) if "True" initialize the unit traces for STDP learning
        :param trace_tc:  Rate of decay of spike trace time constant
        :param rest: rest voltage
        :param reset: post-spike reset voltage
        :param threshold: spike threshold voltage
        :param refractory: post-spike refractory period
        :param voltage_decay: rate of decay of membrane voltage (leakage)
        :return:
        """

        super().__init__()
        self.n = n
        self.rest = rest
        self.threshold = threshold
        self.refractory = refractory
        self.voltage_decay = voltage_decay

        self.v = self.rest * torch.ones(n)  # membrane voltage
        self.s = torch.zeros(n)  # spikes.

        if traces:
            self.x = torch.zeros(n)  # initialize spike traces
            self.trace_tc = trace_tc

        self.refrac_count = torch.zeros(n)  # time counter for refractory time.

    def step(self, inputs, mode, dt):
        """
        On each simulation step, process the input and update traces (x), membrane voltage (v) and spikes (s).
        Also updates internal states, like the unit refractory state.
        :param inputs: vector with the information (i.e. pre_outpur * weight)
        :param mode: 'train' : if is 'train', updates the training traces.
        :param dt: time step
        :return: nothing
        """

        self.v -= dt * self.voltage_decay * (self.v - self.rest)  # update v
        self.refrac_count[self.refrac_count != 0] -= dt   # update refractory state

        # Check for spikes
        self.s = (self.v >= self.threshold) * (self.refrac_count < dt)
        self.refrac_count[self.s] = dt * self.refractory
        self.v[self.s] = self.rest

        if mode == 'train':
            self.x -= dt * self.trace_tc * self.x  # update spike trace
            self.x[self.s.byte()] = 1.0

        # membrane potential update. This update implies that does not affect a spike occurrence, but possible in the
        # next time step, if the voltage contribution implies a level greater than rest + leak
        # == A good question is what if we put it as a first line in this method ===
        # Here, inputs are not a spike train, it's the synaptic output.
        self.v += sum([inputs[key] for key in inputs])

    def get_spikes(self):
        return self.s

    def get_voltages(self):
        return self.s

    def get_traces(self):
        return self.x








