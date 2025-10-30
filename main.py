import torch
import pygame
import numpy as np
from model import Linear_QNet
from snake_game import SnakeGameAI, Direction, Point

# --- Load model ---
MODEL_PATH = "model_final.pth"  # thay bằng file bạn đã lưu
model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# --- Hàm lấy trạng thái giống agent.py ---
def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game._is_collision(point_r)) or
        (dir_l and game._is_collision(point_l)) or
        (dir_u and game._is_collision(point_u)) or
        (dir_d and game._is_collision(point_d)),

        # Danger right
        (dir_u and game._is_collision(point_r)) or
        (dir_d and game._is_collision(point_l)) or
        (dir_l and game._is_collision(point_u)) or
        (dir_r and game._is_collision(point_d)),

        # Danger left
        (dir_d and game._is_collision(point_r)) or
        (dir_u and game._is_collision(point_l)) or
        (dir_r and game._is_collision(point_u)) or
        (dir_l and game._is_collision(point_d)),

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

# --- Chơi game ---
def play_game():
    game = SnakeGameAI(render=True)
    running = True

    while running:
        state = get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = model(state_tensor)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)

        if done:
            game._update_ui()
            pygame.display.flip()
            print(f"Game Over! Score: {score}")

            # Hiển thị thông báo chờ người chơi nhấn phím để chơi lại
            waiting = True
            font = pygame.font.SysFont("arial", 36)
            font = pygame.font.SysFont('arial', 30)

            text_score = font.render(f"Game over", True, (255, 255, 255))
            game.display.blit(text_score, (game.w/2 - 100, game.h/2 - 20))

            pygame.display.update()

            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False
                            game.reset()
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return

if __name__ == "__main__":
    play_game()
