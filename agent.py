import numpy as np
import time


def _get_steps():
    """Returns a list of of all possible moves."""
    #          NW       NE      E       SE      SW        W
    return [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]


def _valid_position(position):
    """Check if a player resides inside the hexagonal playing field.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        position: tuple 
            the coordinates (y, x)

    Returns:
        valid: boolean
            a boolean indicating whether the provided coordinate is valid
    """
    y, x = position
    return (x >= 0 and x < 13 and
            y >= 0 and y < 13 and
            x + y >= 13 // 2 and
            x + y < 13 + 13 // 2)


def _no_crash(board, position):
    """Check if a player's position colides with a tail.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        position: tuple 
            the coordinates (y, x)

    Returns:
        valid: boolean
            a boolean indicating whether the provided coordinate crashes
    """
    y, x = position
    return np.sum(board, axis=2)[y, x] == 0


def terminal(board, position):
    """Check whether a player positioned at position loses.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        position: tuple 
            the coordinates (y, x)

    Returns:
        terminal: boolean
            a boolean indicating whether the game has finished
    """
    return not _valid_position(position) or not _no_crash(board, position)


def _update_pos_orient(position, orientation, action):
    """Update current positition and orientation, given a certain action

    Args:
        position: tuple 
            the coordinates (y, x) of the player
        orientation: int 
            the orientation of the player
        action: int 
            element in [-2, 2]

    Returns:
        new_position: tuple 
            the new coordinates (y, x) of the player
        new_orientation: int 
            the new orientation of the player
    """
    STEPS = _get_steps()
    new_orient = (orientation + action) % 6
    new_position = (position[0] + STEPS[new_orient][0],
                    position[1] + STEPS[new_orient][1])
    return new_position, new_orient


def _update_board(board, position, player):
    """Update current bitmap of a player"""
    board = board.copy()
    board[position[0], position[1], player] = 1
    return board


def get_legal_moves(board, position, orientation):
    """Returns a list of possible legal moves.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        position: tuple 
            the coordinates (y, x) of the player
        orientation: int 
            the orientation of the player

    Returns:
        moves: list
            a list of all legal moves from current position and orientation,
            and their corresponding step.
    """
    STEPS = _get_steps()

    legal_moves = []
    for a in range(-2, 3):
        new_pos, new_orient = _update_pos_orient(position, orientation, a)
        if (_valid_position(new_pos) and _no_crash(board, new_pos)):
            legal_moves.append(a)
    return legal_moves


def hex_distance(position1, position2):
    """Calculate the distance on a hexagonal plane between two (y, x) 
    coordinates in an axial system.
    SOURCE: https://www.redblobgames.com/grids/hexagons/#distances

    Args:
        position1: tuple
            (y, x) coordinates
        position2: tuple
            (y, x) coordinates

    Returns:
        dist: int
            the minimal number of steps needed to go from A to B.

    """
    y1, x1 = position1
    y2, x2 = position2
    return (abs(y1 - y2) + abs(y1 + x1 - y2 - x2) + abs(x1 - x2)) / 2


def distances_all(board, start, orientation, bitmap=None):
    """Calculate the distances from a certain position to all other
    free tiles within the board.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        start: tuple 
            the starting coordinates (y, x) of the player
        orientation: int 
            the orientation of the player
        bitmap: 2D np.ndarray (BOARD_SIZE x BOARD_SIZE)
            indicates if a position is still free

    Returns
        distances: 2D np.ndarray (BOARD_SIZE x BOARD_SIZE)
            the distance to each tile, going from start
    """
    if bitmap is None:
        bitmap = np.sum(board, axis=2)

    distances = np.ones((board.shape[0], board.shape[1]))*100
    distances[start[0], start[1]] = 0
    frontier = {(start, orientation)}

    while len(frontier):
        curr_pos, curr_or = frontier.pop()

        legal_moves = []
        for a in range(-2, 3):
            new_pos, new_or = _update_pos_orient(curr_pos, curr_or, a)
            if _valid_position(new_pos) and bitmap[new_pos[0], new_pos[1]] == 0:
                legal_moves.append((new_pos, new_or))
        
        for new_pos, new_or in legal_moves:
            new_cost = distances[curr_pos[0], curr_pos[1]] + 1
            
            if new_cost < distances[new_pos[0], new_pos[1]]:
                distances[new_pos[0], new_pos[1]] = new_cost
                frontier.add((new_pos, new_or))

    return distances


def utility(board, positions, orientations):
    """Calculate a score, from the perspective of player 1 for each 
    possible game state. First check whether player 1 loses. If so,
    return the total number of unfilled tiles (negative) in order
    to fill up as many tiles as possible before losing. If player 1
    is not losing, we subtract the number of tiles player 2 can reach 
    first from the number of tiles player 1 can reach first.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        positions: iterable of two tuples
            the coordinates (y, x) of both players
        orientations: iterable of two ints
            the orientation of both player

    Returns:
        score: int
            the score for this game state, defined as the difference in
            number of tiles player 1 can reach first and the number of tiles
            player 2 can reach first
        distances: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS)
            the distances to all tiles from their position for each player
    """
    player1_pos, player2_pos = positions
    player1_orient, player2_orient = orientations

    # Check if player 1 is not losing
    if terminal(board, player1_pos):
        # Return a very negative score, since losing is bad mkay
        return 1000 * -127 + np.sum(board)

    # Calculate distances to all tiles for player 1 and 2
    distances = np.ones_like(board)
    distances[:, :, 0] = distances_all(board, player1_pos, player1_orient)
    distances[:, :, 1] = distances_all(board, player2_pos, player2_orient)

    # Calculate difference in number of tiles the players can reach first
    nr_tiles_p1_first = np.sum(distances[:, :, 0] < distances[:, :, 1])
    nr_tiles_p2_first = np.sum(distances[:, :, 0] > distances[:, :, 1])

    return (nr_tiles_p1_first - nr_tiles_p2_first), distances


def make_action(board, positions, orientations, player, action):
    """Rotate according to provided action, and move 1 step forward. 
    Update all data structures afterwards.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        positions: iterable of two tuples
            the coordinates (y, x) of both players
        orientations: iterable of two ints
            the orientation of both player
        player: int
            player index, the index for the final axis in the board
        action: int 
            element in [-2, 2] corresponding to a rotation in 
            range(-120, 180, 60)

    Returns:
        new_board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            updated game board
        new_positions: iterable of two tuples
            updated positions
        new_orientations: iterable of two ints
            updated orientations
    """
    board = _update_board(board, positions[player], player)
    new_pos, new_or = _update_pos_orient(positions[player], orientations[player], action)
    orientations = list(orientations)
    positions = list(positions)
    orientations[player] = new_or
    positions[player] = new_pos
    return board, tuple(positions), tuple(orientations)


def get_nr_moves(board, position, orientation):
    """Get the total number of possible legal moves"""
    return len(get_legal_moves(board, position, orientation))


def generate_move(board, positions, orientations):
    """Generate a move.

    Args:
        board: 3D np.ndarray (BOARD_SIZE x BOARD_SIZE x N_PLAYERS))
            the game board, the final axis contains bitmaps for each player
        position: tuple 
            the coordinates (y, x) of the player
        orientation: int 
            the orientation of the player

    Returns:
        move: int
            an integer in [-2,2]
    """

    # If DEBUG is True, then some information will be printed every round
    DEBUG = True

    # Switch x and y coordinates in the provided positions, such that the
    # y-coordinate is always the first element in the tuple
    positions = [(positions[0][1], positions[0][0]),
                 (positions[1][1], positions[1][0])]

    # We are player 0
    player = 0

    # Start timing how long our move takes
    start = time.time()

    # We will calculate a score for each possible move. We pick the move
    # with the maximum score as our move. In case of a tie, we pick the move
    # that minimizes the total number of possible legal moves after moving,
    # in order to fill up the board as much as possible. In case of a second 
    # tie, we pick the move closest to the center.
    scores = {}

    # Calculate a score for each legal move
    for a in get_legal_moves(board, positions[0], orientations[0]):
        new_state = make_action(board, positions, orientations, player, a)
        new_board, new_positions, new_orientations = new_state

        # Calculate score, number of moves and the distance to middle
        score, distances = utility(new_board, new_positions, new_orientations)
        nr_moves = get_nr_moves(new_board, new_positions[0], new_orientations[0])
        dist_to_mid = distances[6, 6, 0]

        # Avoid head-to-head collissions
        if hex_distance(new_positions[0], new_positions[1]) <= 1:
            score -= 50

        scores[a] = (score, -nr_moves, -dist_to_mid)


    # Pick the move that maximizes the score
    if not scores:
        return 0

    best_move, best_score = sorted(scores.items(), reverse=True,
                                   key=lambda x: x[1])[0]

    if DEBUG:
        ORIENTS = ['NW', 'NE', 'E', 'SE', 'SW', 'W']
        print(
            '', '-'*50, '\n',
            'Round #{}'.format(int(np.sum(board[:, :, 0]))).center(50), '\n',
            '-'*50, '\n',
            'Starting position = {}\n'.format(positions[0]),
            'Starting orientation = {}\n'.format(ORIENTS[orientations[0]]),
            'Moves:'
        )

        for move, (score, nr_moves, dist_to_mid) in scores.items():
            print('\t {:>2} / {:<2}: Score = {:>4} || Moves = {:<4}'.format(
                move, ORIENTS[(orientations[0] + move) % 6], 
                score, nr_moves
            ))

        print(
         ' Best move = {}\n'.format(best_move),
         'New position = {}\n'.format(new_positions[0]),
         'New orient = {}\n'.format(ORIENTS[(orientations[0] + best_move) % 6]),
         'Total time = {}\n\n'.format(np.around(time.time() - start, 4))

        )

    return best_move
