import torch


class Constant:
    def __init__(self, source, target, w=None):
        """
        Specifies constant synapses between two population of neurons
        :param source: pre-synaptic population
        :param target: post-synaptic population
        :param w: weights matrix
        """
        self.source = source
        self.target = target

        if w is None:
            self.w = torch.rand(source.n, target.n)
        elif (source.n == w.shape[0]) & (target.n == w.shape[1]):
            self.w = w
        else:
            raise NameError('BadDimensions')

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        if (self.source.n == w.shape[0]) & (self.target.n == w.shape[1]):
            self.w = w
        else:
            raise NameError('BadDimensions')


class STDP:
    def __int__(self, source, target, w=None, nu_pre=1e-4, nu_post=1e-2, wmax=1.0, norm=78.0):
        """
        Specify STDP-adapted synapses between two population of neurons
        :param source: pre-synaptic population
        :param target: post-synaptic population
        :param w: synaptic weights
        :param nu_pre:
        :param nu_post:
        :param wmax:
        :param norm:
        :return: Nothing
        """
        self.source = source
        self.target = target
        if w is None:
            self.w = torch.rand(source.n, target.n)
        elif (source.n == w.shape[0]) & (target.n == w.shape[1]):
            self.w = w
        else:
            raise NameError('BadDimensions')

        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.wmax = w.max
        self.norm = norm

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        if (self.source.n == w.shape[0]) & (self.target.n == w.shape[1]):
            self.w = w
        else:
            raise NameError('BadDimensions')

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def normalize(self):
        """
        Normiliza vector to have average value 'self.norm'
        """
        self.w *= self.norm / self.w.sum(0).view(1, -1)

    def update(self):
        """
        Perform STDP weight update
        """
        self.w += self.nu_post * (self.source.x.view(self.source.n, 1) * self.target.s.float().view(1, self.target.n))  # Post-synaptic.
        self.w -= self.nu_pre * (self.source.s.float().view(self.source.n, 1) * self.target.x.view(1, self.target.n))  # Pre-synaptic.
        self.w = torch.clamp(self.w, 0, self.wmax)  # Ensure that weights are within [0, self.wmax].

    # The follow need a review, because if you modify the source or the target population, a new synapses definition
    # is needed.

    def set_source(self, source):
        self.source = source

    def set_target(self, target):
        self.target = target
