import numpy as np
import random
import pygame
import sys
import math
from mcts_interface import get_mcts_move, get_mcts_move_with_analysis
from utils import create_board, drop_piece, is_valid_location, get_next_open_row, print_board, winning_move, minimax, draw_board, create_screen

def play_game(model, row_count=6, column_count=7, squaresize=100, blue=(0,0,255), black=(0,0,0),
			   red=(255,0,0), yellow=(255,255,0)):
	width = column_count * squaresize
	radius = int(squaresize/2 - 5)

	PLAYER = 0
	AI = 1

	EMPTY = 0
	PLAYER_PIECE = 1
	AI_PIECE = 2

	board = create_board()
	print_board(board)
	game_over = False

	pygame.init()

	screen = create_screen()
	draw_board(board, PLAYER_PIECE, AI_PIECE, screen)
	pygame.display.update()

	myfont = pygame.font.SysFont("monospace", 75)

	turn = random.randint(PLAYER, AI)

	
	while not game_over:

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

			if event.type == pygame.MOUSEMOTION:
				pygame.draw.rect(screen, black, (0,0, width, squaresize))
				posx = event.pos[0]
				if turn == PLAYER:
					pygame.draw.circle(screen, red, (posx, int(squaresize/2)), radius)

			pygame.display.update()

			if event.type == pygame.MOUSEBUTTONDOWN:
				pygame.draw.rect(screen, black, (0,0, width, squaresize))
				#print(event.pos)
				# Ask for Player 1 Input
				if turn == PLAYER:
					posx = event.pos[0]
					col = int(math.floor(posx/squaresize))

					if is_valid_location(board, col):
						row = get_next_open_row(board, col)
						drop_piece(board, row, col, PLAYER_PIECE)

						if winning_move(board, PLAYER_PIECE):
							label = myfont.render("Player 1 wins!!", 1, red)
							screen.blit(label, (40,10))
							game_over = True

						turn += 1
						turn = turn % 2

						print_board(board)
						draw_board(board, PLAYER_PIECE, AI_PIECE, screen)


		# # Ask for Player 2 Input
		if turn == AI and not game_over:

			#col = random.randint(0, COLUMN_COUNT-1)
			#col = pick_best_move(board, AI_PIECE)
			# col, stats = model(board, time_limit=5)
			col = model(board, time_limit=5)
			# print(f"AI thinking: {stats['iterations']} iterations, {stats['win_rate']:.2f} win rate")

			if is_valid_location(board, col):
				#pygame.time.wait(500)
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, AI_PIECE)

				if winning_move(board, AI_PIECE):
					label = myfont.render("Player 2 wins!!", 1, yellow)
					screen.blit(label, (40,10))
					game_over = True

				print_board(board)
				draw_board(board, PLAYER_PIECE, AI_PIECE, screen)

				turn += 1
				turn = turn % 2

		if game_over:
			pygame.time.wait(3000)
# example
if __name__ == "__main__":
    play_game(get_mcts_move)
