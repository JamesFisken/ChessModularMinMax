import chess
import random
import numpy as np
import pygame
import sys
from pygame.locals import *
import time
from collections import OrderedDict


board = chess.Board()
playing = True

neginf = -1000000000
inf = 1000000000

pygame.init()

fps = 60

rows = 8
columns = 8
squareSize = 70

WHITE = (230, 151, 99)
BLACK = (230, 61, 50)
SELECTED = 35

fpsClock = pygame.time.Clock()
width, height = squareSize*columns, squareSize*rows
screen = pygame.display.set_mode((width, height))

DEFAULT_IMAGE_SIZE = (squareSize, squareSize)
# images
# sample: carImg = pygame.image.load('racecar.png')
Black_Pawn = pygame.image.load('Pieces/Black_Pawn.png')
Black_Knight = pygame.image.load('Pieces/Black_Knight.png')
Black_Bishop = pygame.image.load('Pieces/Black_Bishop.png')
Black_King = pygame.image.load('Pieces/Black_King.png')
Black_Queen = pygame.image.load('Pieces/Black_Queen.png')
Black_Rook = pygame.image.load('Pieces/Black_Rook.png')

White_Pawn = pygame.image.load('Pieces/White_Pawn.png')
White_Knight = pygame.image.load('Pieces/White_Knight.png')
White_Bishop = pygame.image.load('Pieces/White_Bishop.png')
White_King = pygame.image.load('Pieces/White_King.png')
White_Queen = pygame.image.load('Pieces/White_Queen.png')
White_Rook = pygame.image.load('Pieces/White_Rook.png')

Black_Pawn = pygame.transform.scale(Black_Pawn, DEFAULT_IMAGE_SIZE)
Black_Knight = pygame.transform.scale(Black_Knight, DEFAULT_IMAGE_SIZE)
Black_Bishop = pygame.transform.scale(Black_Bishop, DEFAULT_IMAGE_SIZE)
Black_King = pygame.transform.scale(Black_King, DEFAULT_IMAGE_SIZE)
Black_Queen = pygame.transform.scale(Black_Queen, DEFAULT_IMAGE_SIZE)
Black_Rook = pygame.transform.scale(Black_Rook, DEFAULT_IMAGE_SIZE)

White_Pawn = pygame.transform.scale(White_Pawn, DEFAULT_IMAGE_SIZE)
White_Knight = pygame.transform.scale(White_Knight, DEFAULT_IMAGE_SIZE)
White_Bishop = pygame.transform.scale(White_Bishop, DEFAULT_IMAGE_SIZE)
White_King = pygame.transform.scale(White_King, DEFAULT_IMAGE_SIZE)
White_Queen = pygame.transform.scale(White_Queen, DEFAULT_IMAGE_SIZE)
White_Rook = pygame.transform.scale(White_Rook, DEFAULT_IMAGE_SIZE)

def converttosinglenum(x,y):
    return y*8 + x
def converttopygameval(val):

    return x, y


class bot:

    def __init__(self, currentBoard, side):
        self.currentBoard = currentBoard
        self.side = side
        self.piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
        self.piece_attacking_values = {
            chess.Piece(chess.PAWN, chess.WHITE): 100,  # actual 2
            chess.Piece(chess.BISHOP, chess.WHITE): 8,  # actual 13
            chess.Piece(chess.KNIGHT, chess.WHITE): 16,  # actual 8
            chess.Piece(chess.QUEEN, chess.WHITE): 50,  # actual 27
            chess.Piece(chess.KING, chess.WHITE): 1000,  # actual 8
            chess.Piece(chess.ROOK, chess.WHITE): 12,  # actual 14

            chess.Piece(chess.PAWN, chess.BLACK): -100,  # actual 2
            chess.Piece(chess.BISHOP, chess.BLACK): -8,  # actual 13
            chess.Piece(chess.KNIGHT, chess.BLACK): -16,  # actual 8
            chess.Piece(chess.QUEEN, chess.BLACK): -50,  # actual 27
            chess.Piece(chess.KING, chess.BLACK): -1000,  # actual 8
            chess.Piece(chess.ROOK, chess.BLACK): -12  # actual 14
        }

        self.Pawns = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [50, 50, 50, 50, 50, 50, 50, 50],
                          [10, 10, 20, 30, 30, 20, 10, 10],
                          [5, 5, 10, 25, 25, 10, 5, 5],
                          [0, 0, 0, 20, 20, 0, 0, 0],
                          [5, -5, -10, 0, 0, -10, -5, 5],
                          [5, 10, 10, -20, -20, 10, 10, 5],
                          [0, 0, 0, 0, 0, 0, 0, 0]])

        self.FlippedPawns = np.flip(self.Pawns, 0)

        self.Knights = np.array([[-15, -15, -13, -13, -13, -13, -15, -15],
                            [-15, -10, -3, -3, -3, -3, -10, -15],
                            [-13, 0, 6, 9, 9, 6, 0, -13],
                            [-13, 3, 9, 12, 12, 9, 3, -13],
                            [-13, 0, 9, 12, 12, 9, 0, -13],
                            [-13, 3, 6, 9, 9, 6, 3, -13],
                            [-15, -10, -3, 3, 3, -3, -10, -15],
                            [-15, -15, -13, -13, -13, -13, -15, -15]])

        self.FlippedKnights = np.flip(self.Knights, 0)

        self.Bishops = np.array([[-20, -10, -10, -10, -10, -10, -10, -20],
                            [-10, 0, 0, 0, 0, 0, 0, -10],
                            [-10, 0, 5, 10, 10, 5, 0, -10],
                            [-10, 5, 5, 10, 10, 5, 5, -10],
                            [-10, 0, 10, 10, 10, 10, 0, -10],
                            [-10, 10, 10, -20, -20, 10, 10, -10],
                            [-10, 15, 0, 0, 0, 0, 15, -10],
                            [-20, -10, -10, -10, -10, -10, -10, -20]])

        self.FlippedBishops = np.flip(self.Bishops, 0)

        self.Rooks = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [5, 10, 10, 10, 10, 10, 10, 5],
                          [-5, 0, 0, 0, 0, 0, 0, -5],
                          [-5, 0, 0, 0, 0, 0, 0, -5],
                          [-5, 0, 0, 0, 0, 0, 0, -5],
                          [-5, 0, 0, 0, 0, 0, 0, -5],
                          [-5, 0, 0, 0, 0, 0, 0, -5],
                          [0, 0, 0, 5, 5, 0, 0, 0]])

        self.FlippedRooks = np.flip(self.Rooks, 0)

        self.Queens = np.array([[-20, -10, -10, -5, -5, -10, -10, -20],
                           [-10, 0, 0, 0, 0, 0, 0, -10],
                           [-10, 0, 5, 5, 5, 5, 0, -10],
                           [-5, 0, 5, 5, 5, 5, 0, -5],
                           [0, 0, 5, 5, 5, 5, 0, -5],
                           [-10, 5, 5, 5, 5, 5, 0, -10],
                           [-10, 0, 5, 0, 0, 0, 0, -10],
                           [-20, -10, -10, -5, -5, -10, -10, -20]])

        self.FlippedQueens = np.flip(self.Queens, 0)

        self.Kings = np.array([[-30, -40, -40, -50, -50, -40, -40, -30],
                          [-30, -40, -40, -50, -50, -40, -40, -30],
                          [-30, -40, -40, -50, -50, -40, -40, -30],
                          [-30, -40, -40, -50, -50, -40, -40, -30],
                          [-20, -30, -30, -40, -40, -30, -30, -20],
                          [-10, -20, -20, -20, -20, -20, -20, -10],
                          [20, 20, 0, 0, 0, 0, 20, 20],
                          [20, 30, 10, 0, 0, 10, 30, 20]])

        self.FlippedKings = np.flip(self.Kings, 0)

        self.piece_map_valuation = 0.6
        self.piece_usefulness_valuation = 0.3

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or str(board.outcome()) != "None":
            return self.eval(board), None
        moves = list(board.generate_legal_captures()) + list(board.generate_legal_moves())
        moves = OrderedDict.fromkeys(moves)

        if maximizingPlayer:
            max_value = float("-inf")
            for move in moves:
                child = board.copy()
                child.push(move)

                value, _ = self.minimax(child, depth - 1, alpha, beta, False)
                if value > max_value:
                    max_value = value
                    best_move = move
                alpha = max(alpha, max_value)
                if alpha >= beta:
                    break
            return max_value, best_move
        else:
            min_value = float("inf")
            for move in moves:
                child = board.copy()
                child.push(move)

                value, _ = self.minimax(child, depth - 1, alpha, beta, True)
                if value < min_value:
                    min_value = value
                    best_move = move
                beta = min(beta, min_value)
                if alpha >= beta:
                    break
            return min_value, best_move

    def mappiece(self, piece, pieceMap, x, y):
        if piece.islower():
            value = -pieceMap[y, x]*self.piece_map_valuation
        else:
            value = pieceMap[y, x]*self.piece_map_valuation
        return value

    def eval_piece(self):
        pass
    def eval(self, board):
        value = 0
        for square, piece in board.piece_map().items():
            if piece is not None:
                piece_value = self.piece_values.get(piece.symbol().upper())
                if piece_value is not None:
                    if piece.color == chess.WHITE:
                        value += piece_value
                    else:
                        value -= piece_value

                    attacks = len(board.attacks(square))
                    value += attacks / self.piece_attacking_values[piece] * self.piece_usefulness_valuation

                    piece_symbol = piece.symbol()
                    x = chess.square_file(square)
                    y = chess.square_rank(square)
                    '''
                    if piece_symbol == 'p':
                        value += self.mappiece(piece_symbol, self.Pawns, x, y)
                    elif piece_symbol == 'n':
                        value += self.mappiece(piece_symbol, self.Knights, x, y)
                    elif piece_symbol == 'b':
                        value += self.mappiece(piece_symbol, self.Bishops, x, y)
                    elif piece_symbol == 'r':
                        value += self.mappiece(piece_symbol, self.Rooks, x, y)
                    elif piece_symbol == 'q':
                        value += self.mappiece(piece_symbol, self.Queens, x, y)
                    elif piece_symbol == 'k':
                        value += self.mappiece(piece_symbol, self.Kings, x, y)

                    elif piece_symbol == 'P':
                        value += self.mappiece(piece_symbol, self.FlippedPawns, x, y)
                    elif piece_symbol == 'N':
                        value += self.mappiece(piece_symbol, self.FlippedKnights, x, y)
                    elif piece_symbol == 'B':
                        value += self.mappiece(piece_symbol, self.FlippedBishops, x, y)
                    elif piece_symbol == 'R':
                        value += self.mappiece(piece_symbol, self.FlippedRooks, x, y)
                    elif piece_symbol == 'Q':
                        value += self.mappiece(piece_symbol, self.FlippedQueens, x, y)
                    elif piece_symbol == 'K':
                        value += self.mappiece(piece_symbol, self.FlippedKings, x, y)
                    '''
        return value


    def playbestmove(self):
        best_value, best_move = self.minimax(board, 4, float("-inf"), float("inf"), self.side)
        print(best_value, best_move)
        return best_move

    def playrandommove(self):
        legalMoves = list(board.legal_moves)
        move = random.choice(legalMoves)
        return move


player = True  # True white, False black

bot1 = bot(board, not player)
bot1.eval(board)

def draw_board(board):
    board = list(str(board).split())
    board = np.array(board).reshape((8,8))
    #translating board to pygame by flipping and rotating it
    board = np.fliplr(board)
    if player: #if player is white rotate the board 90 degrees else rotate negative 90
        board = np.rot90(board, k=1)
    else:
        board = np.rot90(board, k=-1)
    x = 0
    for x, column in enumerate(board):
        for y, pieceType in enumerate(column):
            pieceType = str(pieceType)
            if (x+y)%2 == 0:
                colour = BLACK
            else:
                colour = WHITE
            pygame.draw.rect(screen, colour,
                             pygame.Rect(x * squareSize, y * squareSize, squareSize, squareSize))

            if pieceType != ".":
                if pieceType == "p":
                    screen.blit(Black_Pawn, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))
                if pieceType == "n":
                    screen.blit(Black_Knight, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))
                if pieceType == "b":
                    screen.blit(Black_Bishop, (x * squareSize, y * squareSize))
                if pieceType == "k":
                    screen.blit(Black_King, (x * squareSize, y * squareSize))
                if pieceType == "q":
                    screen.blit(Black_Queen, (x * squareSize, y * squareSize))
                if pieceType == "r":
                    screen.blit(Black_Rook, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))

                if pieceType == "P":
                    screen.blit(White_Pawn, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))
                if pieceType == "N":
                    screen.blit(White_Knight, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))
                if pieceType == "B":
                    screen.blit(White_Bishop, (x * squareSize, y * squareSize))
                if pieceType == "K":
                    screen.blit(White_King, (x * squareSize, y * squareSize))
                if pieceType == "Q":
                    screen.blit(White_Queen, (x * squareSize, y * squareSize))
                if pieceType == "R":
                    screen.blit(White_Rook, (
                    x * squareSize - 3 * squareSize / 75, y * squareSize + 3 * squareSize / 75))

selected_squares = []
def convert_notation(notation):
    if not player:
        from_square = chess.square(7 - notation[0][0], notation[0][1])
        to_square = chess.square(7 - notation[1][0], notation[1][1])
    else:
        from_square = chess.square(notation[0][0], 7 - notation[0][1])
        to_square = chess.square(notation[1][0], 7 - notation[1][1])
    return chess.Move(from_square, to_square)

def make_move(move):
    print(move)
    if move in board.legal_moves:
        if board.piece_at(move.from_square).piece_type == chess.PAWN and \
                chess.square_rank(move.to_square) == 7:
            if board.turn:
                promoted_piece = 'Q'
            else:
                promoted_piece = 'q'
            move = chess.Move(move.from_square, move.to_square, promotion=chess.Piece.from_symbol(promoted_piece))
            board.push(move)

        else:
            board.push(move)
    else:
        print("Illegal move")
    draw_board(board)

def get_mouse_inputs(x, y, board):
    global selected_squares


    squareX = round((x+squareSize/2)/squareSize)-1
    squareY = round((y+squareSize/2)/squareSize)-1

    if len(selected_squares) == 0:  # first click and clicked on a piece    and boardList[squareX][squareY] != "."
        selected_squares.append([squareX, squareY])

    elif len(selected_squares) == 1:
        selected_squares.append([squareX, squareY])
        print(selected_squares)
        move = convert_notation(selected_squares)
        make_move(move)
        selected_squares = []

start_time = 0
while True:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and board.turn == player:
            x, y = pygame.mouse.get_pos()
            get_mouse_inputs(x, y, board)
            start_time = time.time()

    # Update.

    draw_board(board)
    pygame.display.flip()
    fpsClock.tick(fps)
    if player != board.turn:
        move = bot1.playbestmove()
        board.push(move)
        stop_time = time.time()
        print(stop_time-start_time, " seconds taken")




