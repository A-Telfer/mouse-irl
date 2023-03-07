import gymnasium
import mouse_irl

from pathlib import Path

# Load the data
import mouse_irl
dataset = mouse_irl.datasets.Dataset0()
print(dataset.groups)
data = dataset.load(dataset.find_datafile("saline-saline", "m1"))

# Create the environment
env = gymnasium.make('mouse_irl/OpenFieldEnv-v0')
env.reset(mouse_position=data['initial_position'])

# Run the mouse behaviour in the environment
for action in data['actions']:
    env.step(action)
    env.render()

env.close()
