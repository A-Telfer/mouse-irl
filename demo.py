import gymnasium
import mouse_rl
import json

from pathlib import Path

# Load the data
dataset = Path("./exp0_data")
demo_datafile = dataset / "saline-saline" / "m1.json"
with open(demo_datafile) as fp:
    data = json.load(fp)

# Create the environment
env = gymnasium.make('mouse_rl/OpenFieldEnv-v0')
env.reset(mouse_position=data['initial_position'])

# Run the mouse behaviour in the environment
for action in data['actions']:
    env.step(action)
    env.render()

env.close()
