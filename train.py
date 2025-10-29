# file: train.py
import numpy as np
from agent import Agent
from snake_game import SnakeGameAI

def train():
    agent = Agent()
    game = SnakeGameAI(render = False)
    espisode = 1500
    while agent.n_games <= espisode:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # Train step (và nhận loss)
        loss_value = agent.train_short_memory(state_old, action, reward, state_new, done)

        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score)

    print("Training finished!")
    agent.model.save('model_final.pth')
    
if __name__ == '__main__':
    train()
