import numpy as np
import random
from collections import deque, namedtuple
import time

Direction = namedtuple('Direction', ['dx', 'dy'])
UP = Direction(0, -1)
DOWN = Direction(0, 1)
LEFT = Direction(-1, 0)
RIGHT = Direction(1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
DIR_TO_IDX = {UP:0, RIGHT:1, DOWN:2, LEFT:3}

class SnakeEnv:
    def __init__(self, size=10, max_apples=35, seed=None):
        self.size = size
        self.max_apples = max_apples
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.snake = deque()
        cx = self.size // 2
        cy = self.size // 2
        # Longitud inicial 3, extendiéndose hacia la derecha
        self.snake.appendleft((cx, cy))       # cabeza
        self.snake.append((cx+1, cy))
        self.snake.append((cx+2, cy))         # cola
        self.direction = LEFT
        self.apples_eaten = 0
        self.apple = self._place_apple()
        self.done = False
        self.won = False
        self.steps = 0
        return self._get_obs()

    def _place_apple(self):
        empty = {(x,y) for x in range(self.size) for y in range(self.size)} - set(self.snake)
        if not empty:
            self.apple = None
            self.done = True
            return None
        self.apple = random.choice(list(empty))
        return self.apple

    def _get_obs(self):
        grid = np.zeros((2, self.size, self.size), dtype=np.float32)
        for (x,y) in self.snake:
            grid[0, y, x] = 1.0
        if self.apple is not None:
            ax, ay = self.apple
            grid[1, ay, ax] = 1.0
        return grid

    def available_actions(self):
        acts = []
        for d in DIRECTIONS:
            if (d.dx == -self.direction.dx and d.dy == -self.direction.dy):
                continue
            hx, hy = self.snake[0]
            nx, ny = hx + d.dx, hy + d.dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            if (nx, ny) in set(self.snake) and (nx, ny) != self.snake[-1]:
                continue
            acts.append(d)
        return acts

    def step(self, action_direction):
        if self.done or self.won:
            return self._get_obs(), 0.0, True, {}
        if (action_direction.dx == -self.direction.dx and action_direction.dy == -self.direction.dy):
            action_direction = self.direction
        self.direction = action_direction
        hx, hy = self.snake[0]
        nx, ny = hx + self.direction.dx, hy + self.direction.dy

        # Colisiones
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            self.done = True
            return self._get_obs(), -1.0, True, {'reason':'wall'}
        hit_self = (nx, ny) in set(self.snake)
        grows = (self.apple == (nx, ny))
        if hit_self and not (not grows and (nx, ny) == self.snake[-1]):
            self.done = True
            return self._get_obs(), -1.0, True, {'reason':'self'}

        self.snake.appendleft((nx, ny))
        reward = 0.0
        if grows:
            self.apples_eaten += 1
            if self.apples_eaten >= self.max_apples:
                self.won = True
                reward = 1.0
            else:
                self._place_apple()
                reward = 1.0
        else:
            self.snake.pop()

        self.steps += 1
        return self._get_obs(), reward, self.done or self.won, {}

    def run_episode(self, agent_fn, render=False, vis=None, max_steps=1000, sleep=0.05):
        obs = self.reset()
        start_time = time.perf_counter()
        steps = 0
        while True:
            action = agent_fn(obs, self)
            if action is None:
                reason = 'no_action'
                break
            obs, r, done, info = self.step(action)
            steps += 1
            if render and not vis:
                print(self.render_text())
                print('steps', steps, 'apples', self.apples_eaten)
                print('---')
            if vis:
                vis.draw(self)
            if done:
                reason = info.get('reason','done')
                break
            if steps >= max_steps:
                reason = 'max_steps'
                break
            if sleep>0:
                time.sleep(sleep)
        elapsed = time.perf_counter() - start_time
        return self.apples_eaten, steps, elapsed, reason

    def render_text(self):
        s = [['.' for _ in range(self.size)] for __ in range(self.size)]
        for (x,y) in self.snake:
            s[y][x] = 's'
        hx, hy = self.snake[0]
        s[hy][hx] = 'S'
        if self.apple is not None:
            ax, ay = self.apple
            s[ay][ax] = 'A'
        return '\n'.join(''.join(row) for row in s)
