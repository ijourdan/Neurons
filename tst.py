import numpy as np
import torch

#%%
dt=1
n=10
time_post = 50 * torch.ones(n)
s = torch.zeros(n)
s[5] = 1

#%%
time_post += dt
time_post[time_post>50]=50
time_post[s.bool()] = 0
#%%
s[2] = 0

time_post[time_post<20] += dt
#time_post[time_post>50]=50
time_post[s.bool()] = 0

#%%

