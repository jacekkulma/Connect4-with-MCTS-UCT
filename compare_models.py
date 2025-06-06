import numpy as np
from connect4_model_vs_model import play_game
from utils import minimax

# example
if __name__ == "__main__":
    play_game(minimax, minimax)
