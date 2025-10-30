# file: snake_game.py
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import os

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 30

class SnakeGameAI:
    def __init__(self, w=400, h=300, render = False):
        self.render = render
        if not render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.w = w
        self.h = h
        if render:
            self.display = pygame.display.set_mode((self.w, self.h))
        else:
            self.display = pygame.Surface((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(
            (self.w // (2 * BLOCK_SIZE)) * BLOCK_SIZE,
            (self.h // (2 * BLOCK_SIZE)) * BLOCK_SIZE
        )
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # only process events if rendering (useful for headless training)
        if getattr(self, "render", True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        # small step penalty to encourage shorter paths
        reward = -0.01
        game_over = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            if getattr(self, "render", True):
                self._update_ui()
                pygame.time.delay(100)
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if getattr(self, "render", True):
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hit boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hit itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f"Score: {self.score}", True, (255,255,255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action: [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
