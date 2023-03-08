import gymnasium
import mouse_irl

from pathlib import Path

# Load the data
import mouse_irl

env = gymnasium.make('mouse_irl/OpenFieldEnv-v0', fps=1000)

dataset = mouse_irl.datasets.Dataset0()
for group in dataset:
    for mouse in dataset[group]:
        print("Recording:", group, mouse)
        data = dataset[group, mouse]
        
        # Run the mouse behaviour in the environment
        env.reset(mouse_position=data['initial_position'])
        for action in data['actions']:
            env.step(action)
            env.render()

env.close()
