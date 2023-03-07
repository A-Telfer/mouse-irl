#!/usr/bin/env python
# coding: utf-8

# # Playback Agent

# In[ ]:


import gymnasium
import mouse_irl
import json

from pathlib import Path


# ## Load the agent data

# In[2]:


dataset = mouse_irl.datasets.Dataset0()
data = dataset.load(dataset.find_datafile("saline-saline", "m1"))


# ## Simulate in the RL env

# In[3]:


env = gymnasium.make('mouse_irl/OpenFieldEnv-v0')
env.reset(mouse_position=data['initial_position'])

for i, action in enumerate(data['actions']):
    env.step(action)
    env.render()
    if i > 150:
        break


# In[4]:


env.close()

