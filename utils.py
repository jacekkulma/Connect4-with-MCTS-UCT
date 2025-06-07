import numpy as np
import random
import pygame
import math

def create_screen(row_count=6, column_count=7, squaresize=100):
	width = column_count * squaresize
	height = (row_count+1) * squaresize
	size = (width, height)
	return pygame.display.set_mode(size)

def create_board(row_count=6, column_count=7):
	board = np.zeros((row_count,column_count))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col, row_count=6):
	return board[row_count-1][col] == 0

def get_next_open_row(board, col, row_count=6):
	for r in range(row_count):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece, row_count=6, column_count=7):
	# Check horizontal locations for win
	for c in range(column_count-3):
		for r in range(row_count):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(column_count):
		for r in range(row_count-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(column_count-3):
		for r in range(row_count-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(column_count-3):
		for r in range(3, row_count):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def evaluate_window(window, piece, piece1, piece2, empty):
	score = 0
	opp_piece = piece1
	if piece == piece1:
		opp_piece = piece2

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(empty) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(empty) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(empty) == 1:
		score -= 4

	return score

def score_position(board, piece, piece1, piece2, empty, row_count=6, column_count=7, widow_length=4):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, column_count//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal
	for r in range(row_count):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(column_count-3):
			window = row_array[c:c+widow_length]
			score += evaluate_window(window, piece, piece1, piece2, empty)

	## Score Vertical
	for c in range(column_count):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(row_count-3):
			window = col_array[r:r+widow_length]
			score += evaluate_window(window, piece, piece1, piece2, empty)

	## Score posiive sloped diagonal
	for r in range(row_count-3):
		for c in range(column_count-3):
			window = [board[r+i][c+i] for i in range(widow_length)]
			score += evaluate_window(window, piece, piece1, piece2, empty)

	for r in range(row_count-3):
		for c in range(column_count-3):
			window = [board[r+3-i][c+i] for i in range(widow_length)]
			score += evaluate_window(window, piece, piece1, piece2, empty)

	return score

def is_terminal_node(board, piece1, piece2):
	return winning_move(board, piece1) or winning_move(board, piece2) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer, piece1, piece2, empty):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board, piece1, piece2)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, piece2):
				return (None, 100000000000000)
			elif winning_move(board, piece1):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, piece2, piece1, piece2, empty))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, piece2)
			new_score = minimax(b_copy, depth-1, alpha, beta, False, piece1, piece2, empty)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value

	else: # Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, piece1)
			new_score = minimax(b_copy, depth-1, alpha, beta, True, piece1, piece2, empty)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value

def get_valid_locations(board, column_count=7):
	valid_locations = []
	for col in range(column_count):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations

def pick_best_move(board, piece, piece1, piece2, empty):

	valid_locations = get_valid_locations(board)
	best_score = -10000
	best_col = random.choice(valid_locations)
	for col in valid_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		drop_piece(temp_board, row, col, piece)
		score = score_position(board, piece2, piece1, piece2, empty)
		if score > best_score:
			best_score = score
			best_col = col

	return best_col

def draw_board(board, piece1, piece2, screen, row_count=6, column_count=7, blue=(0,0,255), black=(0,0,0),
			   red=(255,0,0), yellow=(255,255,0), squaresize=100):
	
	radius = int(squaresize/2 - 5)
	height = (row_count+1) * squaresize

	for c in range(column_count):
		for r in range(row_count):
			pygame.draw.rect(screen, blue, (c*squaresize, r*squaresize+squaresize, squaresize, squaresize))
			pygame.draw.circle(screen, black, (int(c*squaresize+squaresize/2), int(r*squaresize+squaresize+squaresize/2)), radius)
	
	for c in range(column_count):
		for r in range(row_count):		
			if board[r][c] == piece1:
				pygame.draw.circle(screen, red, (int(c*squaresize+squaresize/2), height-int(r*squaresize+squaresize/2)), radius)
			elif board[r][c] == piece2: 
				pygame.draw.circle(screen, yellow, (int(c*squaresize+squaresize/2), height-int(r*squaresize+squaresize/2)), radius)
	pygame.display.update()

def get_random_move(board, **kwargs):
	valid_moves = get_valid_locations(board)
	return random.choice(valid_moves)

def is_board_full(board, column_count=7):
    # Assuming BOARD_HEIGHT and BOARD_WIDTH are available globally or passed
    # You might iterate through the top row to see if any spot is empty
    for c in range(column_count):
        if is_valid_location(board, c): # Check if any column still has space
            return False
    return True # No valid locations left, board is full
