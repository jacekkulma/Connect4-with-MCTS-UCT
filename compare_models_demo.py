import numpy as np
from connect4_model_vs_model import play_game
from utils import minimax

# example
if __name__ == "__main__":
    ai1_moves, ai2_moves, winner = play_game(minimax, minimax, console_output=True, pygame_window=True)
    print(f"winner: {winner}, moves: {max(ai1_moves, ai2_moves)}")