import numpy as np
import random
import math
from utils import create_board, drop_piece, is_valid_location, get_next_open_row, print_board, winning_move


# returns [AI1_moves, AI2_moves, winner]
def play_game(model1, model2):
    
    AI1_moves = 0
    AI2_moves = 0

    PLAYER = 0
    AI = 1

    PLAYER_PIECE = 1
    AI_PIECE = 2


    board = create_board()
    print_board(board)
    game_over = False

    turn = random.randint(PLAYER, AI)
        
    while not game_over:
        # AI1 turn
        if turn == PLAYER and not game_over:
            col, model_score = model1(board, 5, -math.inf, math.inf, True)
            col = int(col)

            if is_valid_location(board, col):
                AI1_moves += 1
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    game_over = True
                    return [AI1_moves, AI2_moves, model1.__name__]
        
        if turn == AI and not game_over:
            col, model_score = model2(board, 5, -math.inf, math.inf, True)
            col = int(col)

            if is_valid_location(board, col):
                AI1_moves += 1
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    game_over = True
                    return [AI1_moves, AI2_moves, model2.__name__]
                
                print_board(board)

                turn += 1
                turn = turn % 2
        
        if game_over:
            return