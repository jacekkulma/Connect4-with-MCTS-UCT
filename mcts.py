import numpy as np
import math
import random
import time
from copy import deepcopy

# Constants - should match your main Connect Four file
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""

    def __init__(self, board_state, parent=None, action=None, player=AI_PIECE):
        self.board_state = board_state.copy()
        self.parent = parent
        self.action = action  # The column that led to this state
        self.player = player  # Player who made the move to reach this state
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_valid_moves()

    def get_valid_moves(self):
        """Get list of valid column indices"""
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if self.board_state[ROW_COUNT - 1][col] == 0:  # Top row is empty
                valid_moves.append(col)
        return valid_moves

    def is_fully_expanded(self):
        """Check if all children have been expanded"""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Check if this is a terminal game state"""
        return (self.check_winning_move(PLAYER_PIECE) or
                self.check_winning_move(AI_PIECE) or
                len(self.get_valid_moves()) == 0)

    def check_winning_move(self, piece):
        """Check if the given piece has won - copied from main file logic"""
        board = self.board_state

        # Check horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                        board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True

        # Check vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                        board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True

        # Check positive diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                        board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True

        # Check negative diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                        board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                    return True

        return False

    def get_next_open_row(self, col):
        """Find the next available row in a column"""
        for r in range(ROW_COUNT):
            if self.board_state[r][col] == 0:
                return r
        return None

    def make_move(self, col, piece):
        """Create new board state with the move applied"""
        new_board = self.board_state.copy()
        row = self.get_next_open_row(col)
        if row is not None:
            new_board[row][col] = piece
        return new_board

    def ucb1_value(self, c=math.sqrt(2)):
        """Calculate UCB1 value for node selection"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        """Select child with highest UCB1 value"""
        return max(self.children, key=lambda child: child.ucb1_value())

    def add_child(self, action, next_player):
        """Add a child node for the given action"""
        new_board = self.make_move(action, self.get_opponent(self.player))
        child = MCTSNode(new_board, parent=self, action=action, player=next_player)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child

    def get_opponent(self, player):
        """Get the opponent of the given player"""
        return PLAYER_PIECE if player == AI_PIECE else AI_PIECE

    def update(self, result):
        """Update node statistics with simulation result"""
        self.visits += 1
        self.wins += result


class MCTSConnectFourAI:
    """Monte Carlo Tree Search AI for Connect Four"""

    def __init__(self, simulation_time=1.0, max_iterations=1000):
        self.simulation_time = simulation_time  # Time limit in seconds
        self.max_iterations = max_iterations  # Maximum iterations

    def get_best_move(self, board_state, ai_piece=AI_PIECE):
        """Get the best move using MCTS"""
        root = MCTSNode(board_state, player=ai_piece)

        start_time = time.time()
        iterations = 0

        # Run MCTS iterations
        while (time.time() - start_time < self.simulation_time and
               iterations < self.max_iterations):
            # Selection & Expansion
            node = self.select_and_expand(root)

            # Simulation
            result = self.simulate(node, ai_piece)

            # Backpropagation
            self.backpropagate(node, result)

            iterations += 1

        # Choose best move based on visit count (most robust)
        if not root.children:
            # No children expanded, choose random valid move
            valid_moves = root.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else 0

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def select_and_expand(self, root):
        """Selection and expansion phases of MCTS"""
        node = root

        # Selection: traverse down the tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child()

        # Expansion: add a new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            next_player = node.get_opponent(node.player)
            node = node.add_child(action, next_player)

        return node

    def simulate(self, node, ai_piece):
        """Simulation phase - random playout"""
        board = node.board_state.copy()
        current_player = node.get_opponent(node.player)

        # Random playout until terminal state
        while True:
            # Check for terminal conditions
            if self.is_winning_state(board, PLAYER_PIECE):
                return 0 if ai_piece == AI_PIECE else 1  # Player wins
            elif self.is_winning_state(board, AI_PIECE):
                return 1 if ai_piece == AI_PIECE else 0  # AI wins

            # Get valid moves
            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                return 0.5  # Draw

            # Make random move
            col = random.choice(valid_moves)
            row = self.get_next_open_row(board, col)
            if row is not None:
                board[row][col] = current_player

            # Switch players
            current_player = PLAYER_PIECE if current_player == AI_PIECE else AI_PIECE

    def backpropagate(self, node, result):
        """Backpropagation phase - update all ancestors"""
        while node is not None:
            node.update(result)
            node = node.parent

    def is_winning_state(self, board, piece):
        """Check if the given piece has won"""
        # Check horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                        board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True

        # Check vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                        board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True

        # Check positive diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                        board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True

        # Check negative diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                        board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                    return True

        return False

    def get_valid_moves(self, board):
        """Get list of valid column indices"""
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == 0:
                valid_moves.append(col)
        return valid_moves

    def get_next_open_row(self, board, col):
        """Find the next available row in a column"""
        for r in range(ROW_COUNT):
            if board[r][col] == 0:
                return r
        return None


# Advanced MCTS with additional features
class AdvancedMCTSAI(MCTSConnectFourAI):
    """Enhanced MCTS with UCB1 tuning and statistics"""

    def __init__(self, simulation_time=2.0, max_iterations=2000,
                 ucb_constant=math.sqrt(2)):
        super().__init__(simulation_time, max_iterations)
        self.ucb_constant = ucb_constant

    def get_best_move_with_stats(self, board_state, ai_piece=AI_PIECE):
        """Get best move along with detailed search statistics"""
        root = MCTSNode(board_state, player=ai_piece)
        start_time = time.time()
        iterations = 0

        while (time.time() - start_time < self.simulation_time and
               iterations < self.max_iterations):
            node = self.select_and_expand(root)
            result = self.simulate(node, ai_piece)
            self.backpropagate(node, result)
            iterations += 1

        # Gather detailed statistics
        stats = {
            'iterations': iterations,
            'time_taken': time.time() - start_time,
            'nodes_explored': len(root.children),
            'win_rate': root.wins / root.visits if root.visits > 0 else 0,
            'total_simulations': root.visits
        }

        if not root.children:
            valid_moves = root.get_valid_moves()
            best_move = random.choice(valid_moves) if valid_moves else 0
            stats['fallback_to_random'] = True
        else:
            best_child = max(root.children, key=lambda child: child.visits)
            best_move = best_child.action
            stats['best_move_visits'] = best_child.visits
            stats['best_move_win_rate'] = best_child.wins / best_child.visits
            stats['fallback_to_random'] = False

            # Add information about all possible moves
            stats['move_analysis'] = []
            for child in root.children:
                move_info = {
                    'column': child.action,
                    'visits': child.visits,
                    'win_rate': child.wins / child.visits if child.visits > 0 else 0,
                    'ucb1_value': child.ucb1_value()
                }
                stats['move_analysis'].append(move_info)

        return best_move, stats


if __name__ == "__main__":
    # Test the MCTS implementation
    print("Testing MCTS Connect Four AI Core...")

    # Create empty board
    test_board = np.zeros((ROW_COUNT, COLUMN_COUNT))

    # Test basic MCTS
    print("\n=== Basic MCTS Test ===")
    mcts_ai = MCTSConnectFourAI(simulation_time=0.5)
    best_move = mcts_ai.get_best_move(test_board)
    print(f"MCTS recommends column: {best_move}")

    # Test advanced MCTS with statistics
    print("\n=== Advanced MCTS Test ===")
    advanced_ai = AdvancedMCTSAI(simulation_time=1.0)
    best_move, stats = advanced_ai.get_best_move_with_stats(test_board)

    print(f"Advanced MCTS recommends column: {best_move}")
    print("\nDetailed Search Statistics:")
    for key, value in stats.items():
        if key == 'move_analysis':
            print(f"  {key}:")
            for move in value:
                print(f"    Column {move['column']}: {move['visits']} visits, "
                      f"{move['win_rate']:.3f} win rate, {move['ucb1_value']:.3f} UCB1")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")