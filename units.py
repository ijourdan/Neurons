#%%
import torch
import numpy
from abc import ABC, abstractmethod

#%%

class Group(ABC):
    '''
    Abstract base class for groups of neurons.
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, inpts, mode, dt):
        pass

class InputUnits(Group):
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

class SONGroup(Group):
    '''
    Group of Striatal Neuron leaky integrate-and-fire neurons.
    '''
    def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0,
                 refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
        super().__init__()

        self.n = n  # No. of neurons.
        self.rest = rest  # Rest voltage.
        self.reset = reset  # Post-spike reset voltage.
        self.threshold = threshold  # Spike threshold voltage.
        self.refractory = refractory  # Post-spike refractory period (refractory times dt).
        self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.

        self.v = self.rest * torch.ones(n)  # Neuron voltages.
        self.s = torch.zeros(n)  # Spike occurences.

        if traces:
            self.x = torch.zeros(n)  # Firing traces.
            self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

        self.refrac_count = torch.zeros(n)  # Refractory period counters.

    def step(self, inpts, mode, dt):
        # Decay voltages.
        self.v -= dt * self.voltage_decay * (self.v - self.rest)

        if mode == 'train':
            # Decay spike traces and adaptive thresholds.
            self.x -= dt * self.trace_tc * self.x

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.threshold) * (self.refrac_count == 0)
        self.refrac_count[self.s] = dt * self.refractory
        self.v[self.s] = self.reset

        # Integrate input and decay voltages.
        self.v += sum([inpts[key] for key in inpts])

        if mode == 'train':
            # Setting synaptic traces.
            self.x[self.s.byte()] = 1.0

    def get_spikes(self):
        return self.s

    def get_voltages(self):
        return self.v

    def get_traces(self):
        return self.x




         
 

#%%