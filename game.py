# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

import pygame
import numpy
import random
from collections import namedtuple

# Pygame.
pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)

# Default values.
Point = namedtuple('Point', 'x, y')
snakeColor = (84, 104, 231)
blockSize = 20
speed = 40


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.width = w
        self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = 1
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x-blockSize, self.head.y),
                      Point(self.head.x-(2*blockSize), self.head.y)]

        self.score = 0
        self.food = None
        self.placeFood()

    def placeFood(self):
        x = random.randint(0, (self.width-blockSize)//blockSize)*blockSize
        y = random.randint(0, (self.height-blockSize)//blockSize)*blockSize
        self.food = Point(x, y)
        if self.food in self.snake:
            self.placeFood()

    def update(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.moveSnake(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.placeFood()
        else:
            self.snake.pop()
        self.show()
        self.clock.tick(speed)

        return reward, game_over, self.score

    def collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.width - blockSize or pt.x < 0 or pt.y > self.height - blockSize or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def show(self):
        self.display.fill((255, 255, 255))

        for pt in self.snake:
            pygame.draw.rect(self.display, snakeColor, pygame.Rect(
                pt.x, pt.y, blockSize, blockSize))
            pygame.draw.rect(self.display,  (255, 255, 255),
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, snakeColor, pygame.Rect(
            self.food.x, self.food.y, blockSize, blockSize))
        pygame.draw.rect(self.display,  (255, 255, 255),
                         pygame.Rect(self.food.x+4, self.food.y+4, 12, 12))

        text = font.render("Score " + str(self.score), True,  (0, 0, 0))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def moveSnake(self, action):
        clockWise = [1, 4, 2, 3]
        index = clockWise.index(self.direction)
        direction = self.direction
        if numpy.array_equal(action, [0, 1, 0]):
            direction = clockWise[((index + 1) % 4)]
        elif numpy.array_equal(action, [0, 0, 1]):
            direction = clockWise[((index - 1) % 4)]
        self.direction = direction    
        x = self.head.x
        y = self.head.y
        if direction == 1:
            x += blockSize
        elif direction == 2:
            x -= blockSize
        elif direction == 4:
            y += blockSize
        elif direction == 3:
            y -= blockSize

        self.head = Point(x, y)
