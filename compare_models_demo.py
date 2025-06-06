import numpy as np
from connect4_model_vs_model import play_game
from mcts_interface import get_mcts_move
from utils import get_random_move

# example
if __name__ == "__main__":
    ai1_moves, ai2_moves, winner, winner_alg = play_game(get_mcts_move, get_random_move, console_output=True, pygame_window=True)
    print(f"winner: {winner} - {winner_alg}, moves: {max(ai1_moves, ai2_moves)}")