"""
MCTS Interface for Connect Four Game Integration

This file provides a simple interface to use MCTS AI with your existing Connect Four game.
Simply import get_mcts_move from this file and use it in place of your minimax AI.

Usage in your main Connect Four file:
    from mcts_interface import get_mcts_move

    # Replace your AI move selection with:
    col = get_mcts_move(board, AI_PIECE, time_limit=1.0)
"""

from mcts import MCTSConnectFourAI, AdvancedMCTSAI
import numpy as np

# Game constants (should match your main file)
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2


def get_mcts_move(board, ai_piece=AI_PIECE, time_limit=1.0, difficulty='normal'):
    """
    Get the best move using Monte Carlo Tree Search.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state (6x7 array)
    ai_piece : int
        The piece type for the AI (default: AI_PIECE = 2)
    time_limit : float
        Time limit for MCTS search in seconds (default: 1.0)
    difficulty : str
        Difficulty level: 'easy', 'normal', 'hard', 'expert' (default: 'normal')

    Returns:
    --------
    int
        Column index (0-6) representing the best move
    """

    # Adjust time limits based on difficulty
    difficulty_settings = {
        'easy': {'time': 0.2, 'iterations': 500},
        'normal': {'time': 1.0, 'iterations': 1000},
        'hard': {'time': 2.0, 'iterations': 2000},
        'expert': {'time': 5.0, 'iterations': 5000}
    }

    if difficulty in difficulty_settings:
        settings = difficulty_settings[difficulty]
        time_limit = settings['time']
        max_iterations = settings['iterations']
    else:
        max_iterations = 1000

    # Create MCTS AI instance
    mcts_ai = MCTSConnectFourAI(simulation_time=time_limit, max_iterations=max_iterations)

    # Get and return the best move
    return mcts_ai.get_best_move(board, ai_piece)


def get_mcts_move_with_analysis(board, ai_piece=AI_PIECE, time_limit=2.0):
    """
    Get the best move using MCTS along with detailed analysis.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state
    ai_piece : int
        The piece type for the AI
    time_limit : float
        Time limit for MCTS search in seconds

    Returns:
    --------
    tuple
        (best_column, analysis_dict) where analysis_dict contains:
        - 'iterations': number of MCTS iterations performed
        - 'time_taken': actual time spent searching
        - 'win_rate': estimated win rate for AI from this position
        - 'move_analysis': list of all considered moves with statistics
    """

    advanced_ai = AdvancedMCTSAI(simulation_time=time_limit, max_iterations=3000)
    return advanced_ai.get_best_move_with_stats(board, ai_piece)