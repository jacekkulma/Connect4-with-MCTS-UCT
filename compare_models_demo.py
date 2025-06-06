import numpy as np
from connect4_model_vs_model import play_game
from mcts_interface import get_mcts_move
from mcts_pb_interface import get_mcts_move_progressive_bias, get_mcts_move_progressive_widening, get_mcts_move_dynamic_exploration
from utils import get_random_move

# get_mcts_move - basic mcts
# get_mcts_move_progressive_bias - mcts with progressive bias
# get_mcts_move_progressive_widening - mcts with progressive widening
# get_mcts_move_dynamic_exploration - mcts with progressive bias and widening

# example
if __name__ == "__main__":
    ai1_moves, ai2_moves, winner, winner_alg = play_game(get_mcts_move_progressive_bias, get_mcts_move, console_output=False, pygame_window=True)
    print(f"winner: {winner} - {winner_alg}, moves: {max(ai1_moves, ai2_moves)}")