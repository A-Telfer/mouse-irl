#!/usr/bin/env python
# coding: utf-8

# # Playback Agent

# In[1]:


import gymnasium
import mouse_rl
import json

from pathlib import Path


# ## Load the agent data

# In[2]:


dataset = Path("../exp0_data")
demo_datafile = dataset / "saline-saline" / "m1.json"

with open(demo_datafile) as fp:
    data = json.load(fp)

print(data.keys())


# ## Simulate in the RL env

# In[3]:


env = gymnasium.make('mouse_rl/OpenFieldEnv-v0')
env.reset(mouse_position=data['initial_position'])

for i, action in enumerate(data['actions']):
    env.step(action)
    env.render()
    if i > 150:
        break


# In[4]:


env.close()

