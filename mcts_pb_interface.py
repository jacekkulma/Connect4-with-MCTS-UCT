"""
Enhanced MCTS Interface for Connect Four Game Integration

This file provides interfaces to use various MCTS AI variants with your existing Connect Four game.
Includes standard MCTS, Progressive Bias MCTS, Progressive Widening MCTS, and Dynamic Exploration MCTS.

Usage in your main Connect Four file:
    from enhanced_mcts_interface import (
        get_mcts_move_progressive_bias,
        get_mcts_move_progressive_widening,
        get_mcts_move_dynamic_exploration
    )

    # Progressive Bias variant:
    col = get_mcts_move_progressive_bias(board, AI_PIECE, time_limit=1.0)

    # Progressive Widening variant:
    col = get_mcts_move_progressive_widening(board, AI_PIECE, time_limit=1.0)

    # Dynamic Exploration variant:
    col = get_mcts_move_dynamic_exploration(board, AI_PIECE, time_limit=1.0)
"""

from mcts_pb import (
    ProgressiveBiasMCTS,
    ProgressiveWideningMCTS,
    DynamicExplorationMCTS,
    CombinedMCTS,
    EnhancedMCTSNode
)
import numpy as np
import math

# Game constants (should match your main file)
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2


def get_mcts_move_progressive_bias(board, ai_piece=AI_PIECE, time_limit=2.0,
                                 bias_weight=0.1, difficulty='expert', **kwargs):
    """
    Get the best move using Monte Carlo Tree Search with Progressive Bias.

    Progressive Bias uses domain knowledge (heuristics) to guide the search
    towards more promising moves early in the search process.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state (6x7 array)
    ai_piece : int
        The piece type for the AI (default: AI_PIECE = 2)
    time_limit : float
        Time limit for MCTS search in seconds (default: 2.0)
    bias_weight : float
        Weight of the progressive bias term (default: 0.1)
        Higher values = more influence from heuristics
        Lower values = more exploration
    difficulty : str
        Difficulty level: 'easy', 'normal', 'hard', 'expert'

    Returns:
    --------
    int
        Column index (0-6) representing the best move
    """

    # Adjust parameters based on difficulty
    difficulty_settings = {
        'easy': {'time': 0.5, 'iterations': 500, 'bias': 0.05},
        'normal': {'time': 1.0, 'iterations': 1000, 'bias': 0.1},
        'hard': {'time': 2.0, 'iterations': 2000, 'bias': 0.15},
        'expert': {'time': 4.0, 'iterations': 4000, 'bias': 0.2}
    }

    if difficulty in difficulty_settings:
        settings = difficulty_settings[difficulty]
        time_limit = settings['time']
        max_iterations = settings['iterations']
        bias_weight = settings['bias']
    else:
        max_iterations = 1000

    # Create Progressive Bias MCTS AI instance
    pb_mcts = ProgressiveBiasMCTS(
        simulation_time=time_limit,
        max_iterations=max_iterations,
        bias_weight=bias_weight
    )

    # Get and return the best move
    return pb_mcts.get_best_move(board, ai_piece)


def get_mcts_move_progressive_widening(board, ai_piece=AI_PIECE, time_limit=2.0,
                                     pw_constant=2.0, pw_alpha=0.5,
                                     difficulty='expert', **kwargs):
    """
    Get the best move using Monte Carlo Tree Search with Progressive Widening.

    Progressive Widening controls the rate at which new children are added to nodes,
    focusing computational resources on the most promising branches.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state (6x7 array)
    ai_piece : int
        The piece type for the AI (default: AI_PIECE = 2)
    time_limit : float
        Time limit for MCTS search in seconds (default: 2.0)
    pw_constant : float
        Progressive widening constant C (default: 2.0)
        Higher values = expand children more quickly
    pw_alpha : float
        Progressive widening exponent α (default: 0.5)
        Controls the rate of child expansion: |C(v)| ≤ C * N(v)^α
    difficulty : str
        Difficulty level: 'easy', 'normal', 'hard', 'expert'

    Returns:
    --------
    int
        Column index (0-6) representing the best move
    """

    # Adjust parameters based on difficulty
    difficulty_settings = {
        'easy': {'time': 0.5, 'iterations': 500, 'pw_c': 1.5, 'pw_a': 0.3},
        'normal': {'time': 1.0, 'iterations': 1000, 'pw_c': 2.0, 'pw_a': 0.5},
        'hard': {'time': 2.0, 'iterations': 2000, 'pw_c': 2.5, 'pw_a': 0.6},
        'expert': {'time': 4.0, 'iterations': 4000, 'pw_c': 3.0, 'pw_a': 0.7}
    }

    if difficulty in difficulty_settings:
        settings = difficulty_settings[difficulty]
        time_limit = settings['time']
        max_iterations = settings['iterations']
        pw_constant = settings['pw_c']
        pw_alpha = settings['pw_a']
    else:
        max_iterations = 1000

    # Create Progressive Widening MCTS AI instance
    pw_mcts = ProgressiveWideningMCTS(
        simulation_time=time_limit,
        max_iterations=max_iterations,
        pw_constant=pw_constant,
        pw_alpha=pw_alpha
    )

    # Get and return the best move
    return pw_mcts.get_best_move(board, ai_piece)


def get_mcts_move_dynamic_exploration(board, ai_piece=AI_PIECE, time_limit=2.0,
                                    initial_exploration=None, final_exploration=None,
                                    difficulty='expert', **kwargs):
    """
    Get the best move using Monte Carlo Tree Search with Dynamic Exploration.

    Dynamic Exploration uses a linearly decaying exploration constant in UCB1,
    starting with high exploration and gradually shifting towards exploitation.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state (6x7 array)
    ai_piece : int
        The piece type for the AI (default: AI_PIECE = 2)
    time_limit : float
        Time limit for MCTS search in seconds (default: 2.0)
    initial_exploration : float
        Initial exploration constant (default: sqrt(2) * 2 = ~2.83)
        Higher values = more initial exploration
    final_exploration : float
        Final exploration constant (default: 0.1)
        Lower values = more final exploitation
    difficulty : str
        Difficulty level: 'easy', 'normal', 'hard', 'expert'

    Returns:
    --------
    int
        Column index (0-6) representing the best move
    """

    # Adjust parameters based on difficulty
    difficulty_settings = {
        'easy': {
            'time': 0.5, 'iterations': 500,
            'initial_c': math.sqrt(2) * 1.5, 'final_c': 0.2
        },
        'normal': {
            'time': 1.0, 'iterations': 1000,
            'initial_c': math.sqrt(2) * 2.0, 'final_c': 0.1
        },
        'hard': {
            'time': 2.0, 'iterations': 2000,
            'initial_c': math.sqrt(2) * 2.5, 'final_c': 0.05
        },
        'expert': {
            'time': 4.0, 'iterations': 4000,
            'initial_c': math.sqrt(2) * 3.0, 'final_c': 0.01
        }
    }

    if difficulty in difficulty_settings:
        settings = difficulty_settings[difficulty]
        time_limit = settings['time']
        max_iterations = settings['iterations']
        initial_exploration = initial_exploration or settings['initial_c']
        final_exploration = final_exploration or settings['final_c']
    else:
        max_iterations = 1000
        initial_exploration = initial_exploration or (math.sqrt(2) * 2.0)
        final_exploration = final_exploration or 0.1

    # Create Dynamic Exploration MCTS AI instance
    de_mcts = DynamicExplorationMCTS(
        simulation_time=time_limit,
        max_iterations=max_iterations,
        initial_exploration_constant=initial_exploration,
        final_exploration_constant=final_exploration
    )

    # Get and return the best move
    return de_mcts.get_best_move(board, ai_piece)