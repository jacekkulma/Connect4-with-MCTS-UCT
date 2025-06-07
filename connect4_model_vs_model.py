import numpy as np
import random
import math
import pygame
from utils import create_board, drop_piece, is_valid_location, get_next_open_row, print_board, winning_move, create_screen, draw_board, is_board_full


# returns [AI1_moves, AI2_moves, winner]
def play_game(model1, model2, console_output=True, pygame_window=False, red=(255,0,0), yellow=(255,255,0)):
    
    AI1_moves = 0
    AI2_moves = 0

    AI1 = 0
    AI2 = 1

    EMPTY = 0
    AI1_PIECE = 1
    AI2_PIECE = 2


    board = create_board()

    if console_output:
        print_board(board)

    game_over = False

    if pygame_window:
        pygame.init()
        screen = create_screen()
        draw_board(board, AI1_PIECE, AI2_PIECE, screen)
        pygame.display.update()
        myfont = pygame.font.SysFont("monospace", 75)

    turn = random.randint(AI1, AI2)
        
    while not game_over:
        # AI1 turn
        if turn == AI1 and not game_over:
            # col, model_score = model1(board, 5, -math.inf, math.inf, True, AI1_PIECE, AI2_PIECE, EMPTY)
            col = model1(board, ai_piece=AI1_PIECE, time_limit=1)
            col = int(col)

            if is_valid_location(board, col):
                AI1_moves += 1
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI1_PIECE)

                if winning_move(board, AI1_PIECE):
                    if pygame_window:
                        label = myfont.render("Player 1 wins!!", 1, red)
                        screen.blit(label, (40,10))
                    game_over = True
                    winner = "AI1"
                    winner_alg = model1.__name__
                
                if console_output:
                    print_board(board)
                if pygame_window:
                    draw_board(board, AI1_PIECE, AI2_PIECE, screen)

                turn += 1
                turn = turn % 2
        
        if turn == AI2 and not game_over:
            # col, model_score = model2(board, 5, -math.inf, math.inf, True, AI1_PIECE, AI2_PIECE, EMPTY)
            col = model2(board, ai_piece=AI2_PIECE, time_limit=1)
            col = int(col)

            if is_valid_location(board, col):
                AI1_moves += 1
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI2_PIECE)

                if winning_move(board, AI2_PIECE):
                    if pygame_window:
                        label = myfont.render("Player 2 wins!!", 1, yellow)
                        screen.blit(label, (40,10))
                    game_over = True
                    winner = "AI2"
                    winner_alg = model2.__name__
                
                if console_output:
                    print_board(board)
                if pygame_window:
                    draw_board(board, AI1_PIECE, AI2_PIECE, screen)

                turn += 1
                turn = turn % 2

        if(is_board_full(board) and not game_over):  # You would need to implement this function
            if pygame_window:
                label = myfont.render("It's a Draw!", 1, (0, 0, 255))  # Example color for draw
                screen.blit(label, (40, 10))
            game_over = True
            winner = "Draw"  # Or any other indicator for a draw
            winner_alg = "N/A"  # No specific algorithm won
        
        if game_over:
            if pygame_window:
                pygame.time.wait(3000)
            return [AI1_moves, AI2_moves, winner, winner_alg]