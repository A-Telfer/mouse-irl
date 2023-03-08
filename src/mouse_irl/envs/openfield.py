"""Open field environment for mouse RL.
"""

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class OpenFieldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, width=24, height=19, fps=30, seconds=600, render_mode='human'):
        self.width = width
        self.height = height
        self.fps = fps
        self.seconds = seconds
        self.ticks = self.fps * self.seconds
        # self.tick = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
           "mouse": spaces.Box(
                low=np.array([0,0]), 
                high=np.array([self.width, self.height]), 
                shape=(2,), 
                dtype=int)
        }) 

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Return the observation of the environment."""
        return {"mouse": self._mouse_position}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None, mouse_position=None):
        """Reset the environment."""
        super().reset(seed=seed)

        if mouse_position is None:
            self._mouse_position = np.array([self.width/2, self.height/2]).astype(int)
        else:
            self._mouse_position = np.array(mouse_position).astype(int)

        if self.render_mode == 'human':
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _action_to_delta(self, action):
        """Convert an action to a mouse position delta."""
        if action == 0:
            return np.array([0, -1])
        elif action == 1:
            return np.array([1, 0])
        elif action == 2:
            return np.array([0, 1])
        elif action == 3:
            return np.array([-1, 0])
        elif action == 4:
            return np.array([0, 0])
        else:
            raise ValueError("Invalid action.")

    def step(self, action):
        """Perform a step in the environment."""
        # super().step(action)
        
        self._mouse_position = self._mouse_position + self._action_to_delta(action)
        self._mouse_position[0] = np.clip(self._mouse_position[0], 0, self.width-1)
        self._mouse_position[1] = np.clip(self._mouse_position[1], 0, self.height-1)

        observation = self._get_obs()
        terminated = False
        reward = 0
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        pix_square_size = 20
        window_width = self.width*pix_square_size
        window_height = self.height*pix_square_size

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (window_width, window_height)
            )
            pygame.display.set_caption("Open field")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width*pix_square_size, self.height*pix_square_size))
        canvas.fill((255, 255, 255))
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._mouse_position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (window_width, pix_square_size * y),
                width=3,
            )
        
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()