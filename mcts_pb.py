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

class EnhancedMCTSNode:
    """
    Enhanced MCTS Node with support for progressive bias, widening,
    and a dynamic exploration constant for UCB calculations.
    """

    def __init__(self, board_state, parent=None, action=None, player=AI_PIECE):
        self.board_state = board_state.copy()
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_valid_moves()

        # Progressive bias support
        self.prior_probabilities = {}  # P(s,a) - prior probability for each action
        self.action_values = {}  # Q(s,a) - action values for progressive bias

        # Progressive widening support
        self.children_created = 0
        self.pw_constant = 2.0  # Progressive widening constant
        self.pw_alpha = 0.5  # Progressive widening exponent

    def get_valid_moves(self):
        """
        Get list of valid column indices where a piece can be dropped.
        """
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if self.board_state[ROW_COUNT - 1][col] == 0:
                valid_moves.append(col)
        return valid_moves

    def calculate_prior_probabilities(self):
        """
        Calculate prior probabilities for each valid move based on Connect Four heuristics.
        These probabilities are used for progressive bias.
        """
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return {}

        priors = {}
        total_score = 0

        for col in valid_moves:
            # Heuristic scoring for Connect Four
            score = self.evaluate_move_heuristic(col)
            priors[col] = score
            total_score += score

        # Normalize scores to probabilities
        if total_score > 0:
            for col in priors:
                priors[col] = priors[col] / total_score
        else:
            # If no heuristic preference, distribute uniformly
            prob = 1.0 / len(valid_moves)
            for col in valid_moves:
                priors[col] = prob

        return priors

    def evaluate_move_heuristic(self, col):
        """
        Heuristic evaluation of a potential move.
        This is used to inform the prior probabilities for progressive bias.
        """
        score = 1.0  # Base score

        # Bonus for center columns
        center_bonus = 1.0 + (0.5 - abs(col - 3) / 3.0)
        score *= center_bonus

        # Simulate the move to check for immediate impacts
        temp_board = self.make_move(col, self.player)

        # Bonus for winning moves
        if self.check_winning_move_on_board(temp_board, self.player):
            score *= 100.0

        # Bonus for blocking opponent wins
        opponent = self.get_opponent(self.player)
        temp_board_opponent = self.make_move(col, opponent) # Simulate opponent's potential winning move if current player plays in 'col'
        if self.check_winning_move_on_board(temp_board_opponent, opponent):
            score *= 5.0

        # Bonus for creating multiple threats (3-in-a-row with an empty spot)
        threats = self.count_threats(temp_board, self.player)
        score *= (1.0 + threats * 0.3)

        return score

    def count_threats(self, board, piece):
        """
        Count the number of 3-in-a-row threats for a given piece on the board.
        A threat is a sequence of three pieces with one empty spot that can be filled.
        """
        threats = 0

        # Check horizontal threats
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r][c + i] for i in range(4)]
                if window.count(piece) == 3 and window.count(EMPTY) == 1:
                    threats += 1

        # Check vertical threats
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                window = [board[r + i][c] for i in range(4)]
                if window.count(piece) == 3 and window.count(EMPTY) == 1:
                    threats += 1

        # Check diagonal threats (positive slope)
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r + i][c + i] for i in range(4)]
                if window.count(piece) == 3 and window.count(EMPTY) == 1:
                    threats += 1

        # Check diagonal threats (negative slope)
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r - i][c + i] for i in range(4)]
                if window.count(piece) == 3 and window.count(EMPTY) == 1:
                    threats += 1

        return threats

    def progressive_widening_threshold(self):
        """
        Calculate the dynamic threshold for progressive widening.
        This determines how many children (moves) a node should explore
        based on its visit count.
        """
        if self.visits == 0:
            return 1
        return min(len(self.untried_actions) + len(self.children),
                   int(self.pw_constant * (self.visits ** self.pw_alpha)))

    def should_expand_new_child(self):
        """
        Determines if a new child node should be expanded based on
        the progressive widening threshold.
        """
        return len(self.children) < self.progressive_widening_threshold()

    def ucb1_with_progressive_bias(self, child, c_current, bias_weight=0.1):
        """
        Calculate the UCB1 value for a child node, incorporating progressive bias
        and a dynamic exploration constant (c_current).
        """
        if child.visits == 0:
            return float('inf') # Prioritize unvisited children

        # Standard UCB1 components
        exploitation = child.wins / child.visits
        exploration = c_current * math.sqrt(math.log(self.visits) / child.visits)

        # Progressive bias component, influenced by prior probabilities
        bias = 0.0
        if child.action in self.prior_probabilities:
            prior = self.prior_probabilities[child.action]
            bias = bias_weight * prior * math.sqrt(self.visits) / (1 + child.visits)

        return exploitation + exploration + bias

    def select_child_with_bias(self, c_current, bias_weight):
        """
        Selects the best child node using UCB1 with progressive bias,
        passing the current dynamic exploration constant.
        """
        if not self.prior_probabilities:
            self.prior_probabilities = self.calculate_prior_probabilities()

        return max(self.children,
                   key=lambda child: self.ucb1_with_progressive_bias(child, c_current, bias_weight))

    def select_child_progressive_widening(self, c_current):
        """
        Selects a child based on progressive widening logic.
        If the node should expand a new child, it returns None.
        Otherwise, it selects the best existing child using UCB1 with dynamic c.
        """
        if (self.untried_actions and
                len(self.children) < self.progressive_widening_threshold()):
            return None  # Signal to expand a new child

        if self.children:
            # Select from existing children using standard UCB1 with dynamic c
            return max(self.children, key=lambda child: child.ucb1_value(c_current))

        return None # Should not happen if not terminal and no untried actions

    def ucb1_value(self, c_current):
        """
        Calculate the standard UCB1 value for this node, using a dynamic
        exploration constant (c_current).
        """
        if self.visits == 0:
            return float('inf') # Prioritize unvisited nodes

        exploitation = self.wins / self.visits
        exploration = c_current * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self):
        """
        Check if all possible actions from this node's state have been
        represented by child nodes.
        """
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """
        Check if this game state is a terminal state (win, loss, or draw).
        """
        return (self.check_winning_move(PLAYER_PIECE) or
                self.check_winning_move(AI_PIECE) or
                len(self.get_valid_moves()) == 0)

    def check_winning_move(self, piece):
        """
        Check if the given piece has won on the current board state.
        """
        return self.check_winning_move_on_board(self.board_state, piece)

    def check_winning_move_on_board(self, board, piece):
        """
        Check if the given piece has won on a specified board state.
        Checks horizontal, vertical, and both diagonal directions.
        """
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
        """
        Find the next available row in a given column for placing a piece.
        Returns None if the column is full.
        """
        for r in range(ROW_COUNT):
            if self.board_state[r][col] == EMPTY:
                return r
        return None

    def make_move(self, col, piece):
        """
        Creates a new board state by applying a move (dropping a piece)
        to the current board state.
        """
        new_board = self.board_state.copy()
        row = self.get_next_open_row(col)
        if row is not None:
            new_board[row][col] = piece
        return new_board

    def add_child(self, action, next_player):
        """
        Adds a new child node to the current node, representing the state
        after taking the specified action.
        """
        new_board = self.make_move(action, self.get_opponent(self.player))
        child = EnhancedMCTSNode(new_board, parent=self, action=action, player=next_player)
        self.children.append(child)
        self.untried_actions.remove(action)
        self.children_created += 1
        return child

    def get_opponent(self, player):
        """
        Returns the opponent's piece type for a given player.
        """
        return PLAYER_PIECE if player == AI_PIECE else AI_PIECE

    def update(self, result):
        """
        Updates the node's statistics (visits and wins) based on the
        simulation result.
        """
        self.visits += 1
        self.wins += result

class ProgressiveBiasMCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with Progressive Bias.
    This variant uses heuristic-driven prior probabilities to guide the search.
    """

    def __init__(self, simulation_time=1.0, max_iterations=10000, bias_weight=0.1, exploration_constant=math.sqrt(2)):
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.bias_weight = bias_weight
        self.exploration_constant = exploration_constant # UCB exploration constant

    def find_immediate_win_move(self, board_state, player):
        """
        Check if there's an immediate winning move for the given player.
        Returns the column number of the winning move, or None if no immediate win exists.
        """
        for col in range(COLUMN_COUNT):
            if board_state[ROW_COUNT - 1][col] == EMPTY:  # Column not full
                # Simulate placing the piece
                temp_board = board_state.copy()
                row = self.get_next_open_row(temp_board, col)
                if row is not None:
                    temp_board[row][col] = player
                    if self.is_winning_state(temp_board, player):
                        return col
        return None

    def find_blocking_move(self, board_state, player):
        """
        Check if there's a move needed to block opponent's immediate win.
        Returns the column number of the blocking move, or None if no immediate threat.
        """
        opponent = PLAYER_PIECE if player == AI_PIECE else AI_PIECE
        return self.find_immediate_win_move(board_state, opponent)

    # Modified get_best_move method for any of your MCTS classes:
    def get_best_move(self, board_state, ai_piece=AI_PIECE):
        """
        Enhanced get_best_move that checks for immediate wins/blocks first.
        """
        # Check for immediate winning move
        win_move = self.find_immediate_win_move(board_state, ai_piece)
        if win_move is not None:
            return win_move

        # Check for move to block opponent's immediate win
        block_move = self.find_blocking_move(board_state, ai_piece)
        if block_move is not None:
            return block_move
        """
        Executes the MCTS algorithm to find the best move for the AI.
        """
        root = EnhancedMCTSNode(board_state, player=ai_piece)

        start_time = time.time()
        iterations = 0

        while (time.time() - start_time < self.simulation_time and
               iterations < self.max_iterations):
            # Selection and Expansion phase with progressive bias
            node = self.select_and_expand_with_bias(root)

            # Simulation (playout) phase
            result = self.simulate(node, ai_piece)

            # Backpropagation phase
            self.backpropagate(node, result)

            iterations += 1

        if not root.children:
            valid_moves = root.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else 0

        # After search, choose the child with the most visits (most explored path)
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def select_and_expand_with_bias(self, root):
        """
        Performs the Selection and Expansion phases for Progressive Bias MCTS.
        Traverses the tree, using UCB1 with bias for selection, and expands
        a new child when appropriate.
        """
        node = root

        # Selection: traverse down the tree using UCB1 + progressive bias
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child_with_bias(self.exploration_constant, self.bias_weight)

        # Expansion: if not terminal and not fully expanded, expand a new child
        if not node.is_terminal() and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            next_player = node.get_opponent(node.player)
            node = node.add_child(action, next_player)

        return node

    def simulate(self, node, ai_piece):
        """
        Performs the Simulation (playout) phase.
        Plays out a random game from the current node's state until a terminal state is reached.
        """
        board = deepcopy(node.board_state) # Use deepcopy to avoid modifying original board
        current_player = node.get_opponent(node.player)

        while True:
            # Check for win conditions
            if self.is_winning_state(board, PLAYER_PIECE):
                return 0 if ai_piece == AI_PIECE else 1 # Opponent won
            elif self.is_winning_state(board, AI_PIECE):
                return 1 if ai_piece == AI_PIECE else 0 # AI won

            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                return 0.5 # Draw

            # Make a random valid move
            col = random.choice(valid_moves)
            row = self.get_next_open_row(board, col)
            if row is not None:
                board[row][col] = current_player

            # Switch players
            current_player = PLAYER_PIECE if current_player == AI_PIECE else AI_PIECE

    def backpropagate(self, node, result):
        """
        Performs the Backpropagation phase.
        Updates the visit and win counts for all nodes from the current node up to the root.
        """
        while node is not None:
            node.update(result)
            node = node.parent

    # Helper methods (duplicated from EnhancedMCTSNode for self-containment of simulate, can be refactored)
    def is_winning_state(self, board, piece):
        """Checks if the given piece has won on the provided board."""
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                        board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                        board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                        board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                        board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                    return True
        return False

    def get_valid_moves(self, board):
        """Get list of valid column indices for the provided board."""
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == EMPTY:
                valid_moves.append(col)
        return valid_moves

    def get_next_open_row(self, board, col):
        """Find the next available row in a column for the provided board."""
        for r in range(ROW_COUNT):
            if board[r][col] == EMPTY:
                return r
        return None

class ProgressiveWideningMCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with Progressive Widening.
    This variant controls the rate at which new children are added to nodes.
    """

    def __init__(self, simulation_time=2.0, max_iterations=2000,
                 pw_constant=2.0, pw_alpha=0.5, exploration_constant=math.sqrt(2)):
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.pw_constant = pw_constant
        self.pw_alpha = pw_alpha
        self.exploration_constant = exploration_constant # UCB exploration constant

    def find_immediate_win_move(self, board_state, player):
        """
        Check if there's an immediate winning move for the given player.
        Returns the column number of the winning move, or None if no immediate win exists.
        """
        for col in range(COLUMN_COUNT):
            if board_state[ROW_COUNT - 1][col] == EMPTY:  # Column not full
                # Simulate placing the piece
                temp_board = board_state.copy()
                row = self.get_next_open_row(temp_board, col)
                if row is not None:
                    temp_board[row][col] = player
                    if self.is_winning_state(temp_board, player):
                        return col
        return None

    def find_blocking_move(self, board_state, player):
        """
        Check if there's a move needed to block opponent's immediate win.
        Returns the column number of the blocking move, or None if no immediate threat.
        """
        opponent = PLAYER_PIECE if player == AI_PIECE else AI_PIECE
        return self.find_immediate_win_move(board_state, opponent)

    # Modified get_best_move method for any of your MCTS classes:
    def get_best_move(self, board_state, ai_piece=AI_PIECE):
        """
        Enhanced get_best_move that checks for immediate wins/blocks first.
        """
        # Check for immediate winning move
        win_move = self.find_immediate_win_move(board_state, ai_piece)
        if win_move is not None:
            return win_move

        # Check for move to block opponent's immediate win
        block_move = self.find_blocking_move(board_state, ai_piece)
        if block_move is not None:
            return block_move
        """
        Executes the MCTS algorithm to find the best move for the AI.
        """
        root = EnhancedMCTSNode(board_state, player=ai_piece)
        root.pw_constant = self.pw_constant
        root.pw_alpha = self.pw_alpha

        start_time = time.time()
        iterations = 0

        while (time.time() - start_time < self.simulation_time and
               iterations < self.max_iterations):
            # Selection and Expansion phase with progressive widening
            node = self.select_and_expand_with_widening(root)

            # Simulation (playout) phase
            result = self.simulate(node, ai_piece)

            # Backpropagation phase
            self.backpropagate(node, result)

            iterations += 1

        if not root.children:
            valid_moves = root.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else 0

        # After search, choose the child with the most visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def select_and_expand_with_widening(self, root):
        """
        Performs the Selection and Expansion phases for Progressive Widening MCTS.
        Traverses the tree, using UCB1 for selection, and expands new children
        based on the progressive widening threshold.
        """
        node = root

        while not node.is_terminal():
            # Check if we should expand a new child based on progressive widening
            selected_child = node.select_child_progressive_widening(self.exploration_constant)

            if selected_child is None:
                # Time to expand a new child
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    next_player = node.get_opponent(node.player)
                    node = node.add_child(action, next_player)
                    # Ensure progressive widening parameters are passed to the new child
                    node.pw_constant = self.pw_constant
                    node.pw_alpha = self.pw_alpha
                break # Break after expanding a new child, it's now ready for simulation
            else:
                node = selected_child # Continue selection down the tree

        return node

    def simulate(self, node, ai_piece):
        """
        Performs the Simulation (playout) phase with random playout.
        """
        board = deepcopy(node.board_state)
        current_player = node.get_opponent(node.player)

        while True:
            if self.is_winning_state(board, PLAYER_PIECE):
                return 0 if ai_piece == AI_PIECE else 1
            elif self.is_winning_state(board, AI_PIECE):
                return 1 if ai_piece == AI_PIECE else 0

            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                return 0.5

            col = random.choice(valid_moves)
            row = self.get_next_open_row(board, col)
            if row is not None:
                board[row][col] = current_player

            current_player = PLAYER_PIECE if current_player == AI_PIECE else AI_PIECE

    def backpropagate(self, node, result):
        """
        Performs the Backpropagation phase.
        """
        while node is not None:
            node.update(result)
            node = node.parent

    # Helper methods (duplicated for self-containment)
    def is_winning_state(self, board, piece):
        """Checks if the given piece has won on the provided board."""
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                        board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                        board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                        board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                        board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                    return True
        return False

    def get_valid_moves(self, board):
        """Get list of valid column indices for the provided board."""
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == EMPTY:
                valid_moves.append(col)
        return valid_moves

    def get_next_open_row(self, board, col):
        """Find the next available row in a column for the provided board."""
        for r in range(ROW_COUNT):
            if board[r][col] == EMPTY:
                return r
        return None

class DynamicExplorationMCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with Dynamic Exploration.
    The exploration constant (c in UCB1) decreases linearly over time,
    promoting more exploration early and more exploitation later in the search.
    """

    def __init__(self, simulation_time=2.0, max_iterations=2000,
                 initial_exploration_constant=math.sqrt(2), final_exploration_constant=0.1):
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.initial_c = initial_exploration_constant
        self.final_c = final_exploration_constant

    def find_immediate_win_move(self, board_state, player):
        """
        Check if there's an immediate winning move for the given player.
        Returns the column number of the winning move, or None if no immediate win exists.
        """
        for col in range(COLUMN_COUNT):
            if board_state[ROW_COUNT - 1][col] == EMPTY:  # Column not full
                # Simulate placing the piece
                temp_board = board_state.copy()
                row = self.get_next_open_row(temp_board, col)
                if row is not None:
                    temp_board[row][col] = player
                    if self.is_winning_state(temp_board, player):
                        return col
        return None

    def find_blocking_move(self, board_state, player):
        """
        Check if there's a move needed to block opponent's immediate win.
        Returns the column number of the blocking move, or None if no immediate threat.
        """
        opponent = PLAYER_PIECE if player == AI_PIECE else AI_PIECE
        return self.find_immediate_win_move(board_state, opponent)

    # Modified get_best_move method for any of your MCTS classes:
    def get_best_move(self, board_state, ai_piece=AI_PIECE):
        """
        Enhanced get_best_move that checks for immediate wins/blocks first.
        """
        # Check for immediate winning move
        win_move = self.find_immediate_win_move(board_state, ai_piece)
        if win_move is not None:
            return win_move

        # Check for move to block opponent's immediate win
        block_move = self.find_blocking_move(board_state, ai_piece)
        if block_move is not None:
            return block_move
        """
        Executes the MCTS algorithm to find the best move for the AI,
        with a dynamically decaying exploration constant.
        """
        root = EnhancedMCTSNode(board_state, player=ai_piece)

        start_time = time.time()
        iterations = 0

        while (time.time() - start_time < self.simulation_time and
               iterations < self.max_iterations):
            elapsed_time = time.time() - start_time
            # Calculate current exploration constant based on linear decay over time
            total_time = self.simulation_time if self.simulation_time > 0 else 1.0 # Avoid division by zero
            c_current = self.initial_c - (self.initial_c - self.final_c) * (elapsed_time / total_time)
            c_current = max(c_current, self.final_c) # Ensure c_current does not go below final_c

            # Selection phase: traverse down using UCB1 with the dynamically calculated c_current
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                # This variant uses standard UCB1 for selection, but with a dynamic 'c'
                # If a child has 0 visits, UCB1 will return infinity, ensuring it's selected.
                node = max(node.children, key=lambda child: child.ucb1_value(c_current))

            # Expansion phase: expand a new child if the current node is not terminal and has untried actions
            if not node.is_terminal() and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                next_player = node.get_opponent(node.player)
                node = node.add_child(action, next_player)

            # Simulation (playout) phase
            result = self.simulate(node, ai_piece)

            # Backpropagation phase
            self.backpropagate(node, result)

            iterations += 1

        if not root.children:
            valid_moves = root.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else 0

        # After search, choose the child with the most visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def simulate(self, node, ai_piece):
        board = deepcopy(node.board_state)
        current_player = node.player  # FIXED: Start with current player

        while True:
            if self.is_winning_state(board, PLAYER_PIECE):
                return 0 if ai_piece == AI_PIECE else 1
            elif self.is_winning_state(board, AI_PIECE):
                return 1 if ai_piece == AI_PIECE else 0

            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                return 0.5

            col = random.choice(valid_moves)
            row = self.get_next_open_row(board, col)
            if row is not None:
                board[row][col] = current_player

            current_player = PLAYER_PIECE if current_player == AI_PIECE else AI_PIECE

    def backpropagate(self, node, result):
        """
        Performs the Backpropagation phase.
        """
        while node is not None:
            node.update(result)
            node = node.parent

    # Helper methods (duplicated for self-containment)
    def is_winning_state(self, board, piece):
        """Checks if the given piece has won on the provided board."""
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                        board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                        board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                        board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                        board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                    return True
        return False

    def get_valid_moves(self, board):
        """Get list of valid column indices for the provided board."""
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == EMPTY:
                valid_moves.append(col)
        return valid_moves

    def get_next_open_row(self, board, col):
        """Find the next available row in a column for the provided board."""
        for r in range(ROW_COUNT):
            if board[r][col] == EMPTY:
                return r
        return None