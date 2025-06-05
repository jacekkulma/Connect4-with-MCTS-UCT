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


def get_quick_mcts_move(board, ai_piece=AI_PIECE):
    """
    Get a quick MCTS move for faster gameplay (0.3 second limit).

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state
    ai_piece : int
        The piece type for the AI

    Returns:
    --------
    int
        Column index representing the best move
    """
    return get_mcts_move(board, ai_piece, time_limit=0.3, difficulty='easy')


def compare_mcts_vs_random(board, ai_piece=AI_PIECE, num_simulations=100):
    """
    Compare MCTS move selection against random moves (for testing).

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state
    ai_piece : int
        The piece type for the AI
    num_simulations : int
        Number of random comparisons to make

    Returns:
    --------
    dict
        Comparison statistics
    """
    import random

    # Get MCTS move
    mcts_move = get_mcts_move(board, ai_piece, time_limit=0.5)

    # Get valid moves
    valid_moves = []
    for col in range(7):  # COLUMN_COUNT
        if board[5][col] == 0:  # ROW_COUNT-1, top row is empty
            valid_moves.append(col)

    if not valid_moves:
        return {'error': 'No valid moves available'}

    # Count how often random would pick the same move
    random_matches = 0
    for _ in range(num_simulations):
        random_move = random.choice(valid_moves)
        if random_move == mcts_move:
            random_matches += 1

    return {
        'mcts_move': mcts_move,
        'valid_moves': valid_moves,
        'random_match_rate': random_matches / num_simulations,
        'mcts_vs_random_difference': 1 - (1 / len(valid_moves))  # Expected difference
    }


# Convenience function for different AI personalities
def get_ai_move(board, personality='balanced', ai_piece=AI_PIECE):
    """
    Get AI move based on different personalities.

    Parameters:
    -----------
    board : numpy.ndarray
        The current game board state
    personality : str
        AI personality: 'aggressive', 'balanced', 'defensive', 'quick'
    ai_piece : int
        The piece type for the AI

    Returns:
    --------
    int
        Column index representing the best move
    """

    personality_configs = {
        'aggressive': {'time_limit': 3.0, 'difficulty': 'expert'},
        'balanced': {'time_limit': 1.5, 'difficulty': 'hard'},
        'defensive': {'time_limit': 2.0, 'difficulty': 'hard'},
        'quick': {'time_limit': 0.5, 'difficulty': 'normal'}
    }

    config = personality_configs.get(personality, personality_configs['balanced'])
    return get_mcts_move(board, ai_piece, config['time_limit'], config['difficulty'])


# Testing and demonstration functions
if __name__ == "__main__":
    print("MCTS Interface Testing")
    print("=" * 50)

    # Create a test board
    test_board = np.zeros((6, 7))

    # Test basic functionality
    print("\n1. Basic MCTS Move:")
    move = get_mcts_move(test_board)
    print(f"   Recommended column: {move}")

    print("\n2. Different Difficulty Levels:")
    difficulties = ['easy', 'normal', 'hard', 'expert']
    for diff in difficulties:
        move = get_mcts_move(test_board, difficulty=diff)
        print(f"   {diff.capitalize()}: Column {move}")

    print("\n3. AI Personalities:")
    personalities = ['quick', 'balanced', 'aggressive', 'defensive']
    for personality in personalities:
        move = get_ai_move(test_board, personality=personality)
        print(f"   {personality.capitalize()}: Column {move}")

    print("\n4. Detailed Analysis:")
    move, analysis = get_mcts_move_with_analysis(test_board, time_limit=1.0)
    print(f"   Best move: Column {move}")
    print(f"   Iterations: {analysis['iterations']}")
    print(f"   Time taken: {analysis['time_taken']:.3f}s")
    print(f"   Win rate: {analysis['win_rate']:.3f}")

    if 'move_analysis' in analysis:
        print("   Move breakdown:")
        for move_info in analysis['move_analysis'][:3]:  # Show top 3
            print(f"     Column {move_info['column']}: "
                  f"{move_info['visits']} visits, "
                  f"{move_info['win_rate']:.3f} win rate")

    print("\n5. Comparison with Random:")
    comparison = compare_mcts_vs_random(test_board)
    print(f"   MCTS choice: Column {comparison['mcts_move']}")
    print(f"   Valid moves: {comparison['valid_moves']}")
    print(f"   Random match rate: {comparison['random_match_rate']:.3f}")

    print("\nMCTS Interface ready for integration!")
    print("Import with: from mcts_interface import get_mcts_move")