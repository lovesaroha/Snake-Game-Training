# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Point
from model import LinearQNet, QTrainer

maxMemory = 100_000
batchSize = 1000
learningRate = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9 
        self.memory = deque(maxlen=maxMemory) 
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=learningRate, gamma=self.gamma)


    def getState(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == 2
        dir_r = game.direction == 1
        dir_u = game.direction == 3
        dir_d = game.direction == 4

        state = [
            (dir_r and game.collision(point_r)) or 
            (dir_l and game.collision(point_l)) or 
            (dir_u and game.collision(point_u)) or 
            (dir_d and game.collision(point_d)),
            (dir_u and game.collision(point_r)) or 
            (dir_d and game.collision(point_l)) or 
            (dir_l and game.collision(point_u)) or 
            (dir_r and game.collision(point_d)),
            (dir_d and game.collision(point_r)) or 
            (dir_u and game.collision(point_l)) or 
            (dir_r and game.collision(point_u)) or 
            (dir_l and game.collision(point_d)),
          
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,  
            game.food.x > game.head.x, 
            game.food.y < game.head.y,  
            game.food.y > game.head.y 
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def trainLongMemory(self):
        if len(self.memory) > batchSize:
            mini_sample = random.sample(self.memory, batchSize) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)

    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        state_old = agent.getState(game)
        final_move = agent.getAction(state_old)
        reward, done, score = game.update(final_move)
        state_new = agent.getState(game)
        agent.trainShortMemory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.trainLongMemory()

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()