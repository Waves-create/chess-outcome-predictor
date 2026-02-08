#For Chess logic and feature extraction

import chess

# Material balance feature

def material_balance(board):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    score = 0
    for piece, val in values.items():
        score += len(board.pieces(piece, chess.WHITE)) * val
        score -= len(board.pieces(piece, chess.BLACK)) * val

    return score



# Extract features from moves

def extract_features_from_moves(moves_str, max_moves=40):

    board = chess.Board()

    captures = 0
    checks = 0
    move_count = 0

    moves = moves_str.split()

    for san in moves:

        try:
            move = board.parse_san(san)
        except:
            break

        if board.is_capture(move):
            captures += 1

        board.push(move)

        if board.is_check():
            checks += 1

        move_count += 1

        if max_moves and move_count >= max_moves:
            break

    return {
        "captures": captures,
        "checks": checks,
        "turns_played": move_count,
        "material_balance": material_balance(board)
    }
