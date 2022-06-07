import copy
import time
import abc
import random


class Game(object):
    """A connect four game."""

    def __init__(self, grid):
        """Instances differ by their board."""
        self.grid = copy.deepcopy(grid)  # No aliasing!

    def display(self):
        """Print the game board."""
        for row in self.grid:
            for mark in row:
                print(mark, end='')
            print()
        print()

    def is_col_avail(self, col_num):
        """Checks is a column in the connect 4 board is available as a move."""
        for row_num in range(len(self.grid)):
            if self.grid[row_num][col_num] == '-':
                return True

        return False

    def get_min_avail_space(self, col_num):
        """Returns the minimum (highest index) position in a column that is empty."""
        for row_num in range(len(self.grid)-1,-1,-1):
            if self.grid[row_num][col_num] == '-':
                return row_num

        return None

    def possible_moves(self):
        """Return a list of possible moves given the current board."""
        # YOU FILL THIS IN
        moves = []
        for row_num in range(len(self.grid[0])):
            if self.is_col_avail(row_num):
                moves.append(row_num)

        return moves

    def order_moves(self, move_list):
        """Sorts in-place a list of possible moves with the bad moves appearing before the good moves."""
        median = len(self.grid[0]) // 2
        move_list.sort(key=lambda x: abs(median-x))

    def neighbor(self, col, color):
        """Return a Game instance like this one but with a move made into the specified column."""
        # YOU FILL THIS IN
        new_grid = copy.deepcopy(self.grid)
        row = self.get_min_avail_space(col)
        if new_grid[row][col] == '-':
            new_grid[row][col] = color

        return Game(new_grid)

    def get_grid_mirror_symmetry(self):
        """Return the vertical mirror symmetry of the game's grid."""
        new_grid = copy.deepcopy(self.grid)
        for i in range(len(new_grid)):
            new_grid[i] = new_grid[i][::-1]

        return new_grid

    def get_util_sign(self, color):
        """Returns the multiplicative sign for the given color red or black."""
        if color == 'R':
            return 1
        elif color == 'B':
            return -1
        else:
            return 0

    def get_other_color(self, color):
        """Given a color, returns the opponent's color."""
        if color == 'R':
            return 'B'
        elif color == 'B':
            return 'R'
        else:
            return '-'

    def utility(self):
        """Return the minimax utility value of this game"""
        # YOU FILL THIS IN
        if (win_state := self.winning_state()) is not None:
            return win_state

        util_val = 0
        COINS_NEEDED = 4
        row_util_grid = copy.deepcopy(self.grid)  # not the same as electrical grid!
        col_util_grid = copy.deepcopy(self.grid)
        left_diag_util_grid = copy.deepcopy(self.grid)
        right_diag_util_grid = copy.deepcopy(self.grid)

        # For each cell, there are 7 possible directions:
        # N, E, W, NE, NW, SE, SW
        for i in range(len(self.grid)-1,-1,-1): # Start from the bottom up.
            for j in range(len(self.grid[0])):
                # Start from a coin and add the utility values for all 4 orientations.
                if self.grid[i][j] != '-':
                    curr_color = self.grid[i][j]
                    other_color = self.get_other_color(curr_color)
                    util_sign = self.get_util_sign(curr_color) # to decide whether to +/- to/from the total utility.
                    coin_count = 0
                    space_count = 0

                    # Check columns.
                    if col_util_grid[i][j] != '#':
                        # Go North
                        curr_i = i
                        while curr_i >= 0 and col_util_grid[curr_i][j] != other_color:
                            if col_util_grid[curr_i][j] == curr_color:
                                coin_count += 1
                                col_util_grid[curr_i][j] = '#'  # visited coins are marked with a '#'.
                            elif col_util_grid[curr_i][j] == '-':
                                space_count += 1
                                if space_count > 2:
                                    break  # there can be no coins above an empty space '-'.
                            curr_i -= 1

                        # There can be no spaces under a coin.
                        # So we don't need to check the South direction.

                        if coin_count + space_count >= COINS_NEEDED:
                            util_val += coin_count * util_sign * 2  # give more weight to connections.

                        # Reset values
                        coin_count = 0
                        space_count = 0

                    # Check rows.
                    if row_util_grid[i][j] != '#':
                        # Go East
                        curr_j = j
                        while curr_j < len(self.grid[0]) and row_util_grid[i][curr_j] != other_color:
                            if row_util_grid[i][curr_j] == curr_color:
                                coin_count += 1
                                row_util_grid[i][curr_j] = '#'
                            elif row_util_grid[i][curr_j] == '-':
                                space_count += 1
                            curr_j += 1

                        # Go West
                        curr_j = j - 1
                        while curr_j >= 0 and row_util_grid[i][curr_j] != other_color:
                            if row_util_grid[i][curr_j] == curr_color:
                                coin_count += 1
                                row_util_grid[i][curr_j] = '#'
                            elif row_util_grid[i][curr_j] == '-':
                                space_count += 1
                            curr_j -= 1

                        if coin_count + space_count >= COINS_NEEDED:
                            util_val += coin_count * util_sign * 2

                        # Reset values
                        coin_count = 0
                        space_count = 0

                    # NE, NW, SE, SW:

                    # Check left diagonals.
                    if left_diag_util_grid[i][j] != '#':
                        # Go Northwest
                        curr_i = i
                        curr_j = j
                        util_sign = self.get_util_sign(curr_color)
                        while curr_i >= 0 and curr_j >= 0 and left_diag_util_grid[curr_i][curr_j] != other_color:
                            if left_diag_util_grid[curr_i][curr_j] == curr_color:
                                coin_count += 1
                                left_diag_util_grid[curr_i][curr_j] = '#'
                            elif left_diag_util_grid[curr_i][curr_j] == '-':
                                space_count += 1
                            curr_i -= 1
                            curr_j -= 1

                        # Go Southeast
                        curr_i = i + 1
                        curr_j = j + 1
                        while curr_i < len(self.grid) and curr_j < len(self.grid[0]) and \
                                left_diag_util_grid[curr_i][curr_j] != other_color:
                            if left_diag_util_grid[curr_i][curr_j] == curr_color:
                                coin_count += 1
                                left_diag_util_grid[curr_i][curr_j] = '#'
                            elif left_diag_util_grid[curr_i][curr_j] == '-':
                                space_count += 1
                            curr_i += 1
                            curr_j += 1

                        if coin_count + space_count >= COINS_NEEDED:
                            util_val += coin_count * util_sign * 2

                        # Reset values
                        coin_count = 0
                        space_count = 0

                    # Check right diagonals.
                    if right_diag_util_grid[i][j] != '#':
                        # Go Northeast
                        curr_i = i
                        curr_j = j
                        while curr_i >= 0 and curr_j < len(self.grid[0]) and \
                                right_diag_util_grid[curr_i][curr_j] != other_color:
                            if right_diag_util_grid[curr_i][curr_j] == curr_color:
                                coin_count += 1
                                right_diag_util_grid[curr_i][curr_j] = '#'
                            elif right_diag_util_grid[curr_i][curr_j] == '-':
                                space_count += 1
                            curr_i -= 1
                            curr_j += 1

                        # Go Southwest
                        curr_i = i + 1
                        curr_j = j - 1
                        while curr_i < len(self.grid) and curr_j >= 0 and \
                                right_diag_util_grid[curr_i][curr_j] != other_color:
                            if right_diag_util_grid[curr_i][curr_j] == curr_color:
                                coin_count += 1
                                right_diag_util_grid[curr_i][curr_j] = '#'
                            elif right_diag_util_grid[curr_i][curr_j] == '-':
                                space_count += 1
                            curr_i += 1
                            curr_j -= 1

                        if coin_count + space_count >= COINS_NEEDED:
                            util_val += coin_count * util_sign * 2

        return util_val

    # def is_first_players_turn(self):
    #     """"Checks the board to see if it is the first or second player's turn."""
    #     red_coin_count = 0
    #     black_coin_count = 0
    #     for i in range(len(self.grid[0])):
    #         for coin in self.grid[i]:
    #             if coin == 'R':
    #                 red_coin_count += 1
    #             elif coin == 'B':
    #                 black_coin_count += 1
    #
    #     return red_coin_count == black_coin_count

    def is_board_full(self):
        """"Checks if the connect 4 board is full."""
        for col_num in range(len(self.grid[0])):
            if self.is_col_avail(col_num):
                return False

        return True

    def winning_state(self):
        """Returns float("inf") if Red wins; float("-inf") if Black wins;
           0 if board full; None if not full and no winner"""
        # YOU FILL THIS IN
        def check_rows():
            """Checks the rows of the board to see if there are 4 consecutive coins."""
            for row in self.grid:
                red_seq_count = 0
                black_seq_count = 0
                for coin in row:
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")

            return None

        def check_cols():
            """Checks the columns of the board to see if there are 4 consecutive coins."""
            for col_i in range(len(self.grid[0])):
                red_seq_count = 0
                black_seq_count = 0
                for row_i in range(len(self.grid)):
                    coin = self.grid[row_i][col_i]
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")

            return None

        def check_diagonals():
            """Checks the diagonals of the board to see if there are 4 consecutive coins."""
            # Check left diagonals.
            for i in range(len(self.grid)-3):
                row_i = i
                col_i = 0
                red_seq_count = 0
                black_seq_count = 0
                while row_i < len(self.grid) and col_i < len(self.grid[0]):
                    coin = self.grid[row_i][col_i]
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")
                    row_i += 1
                    col_i += 1

            for j in range(1, len(self.grid[0])-3):
                row_i = 0
                col_j = j
                red_seq_count = 0
                black_seq_count = 0
                while row_i < len(self.grid) and col_j < len(self.grid[0]):
                    coin = self.grid[row_i][col_j]
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")
                    row_i += 1
                    col_j += 1

            # Check right diagonals.
            for i in range(3, len(self.grid)):
                row_i = i
                col_j = 0
                red_seq_count = 0
                black_seq_count = 0
                while row_i >= 0 and col_j < len(self.grid[0]):
                    coin = self.grid[row_i][col_j]
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")
                    row_i -= 1
                    col_j += 1

            for j in range(1, len(self.grid[0]) - 3):
                row_i = len(self.grid) - 1
                col_j = j
                red_seq_count = 0
                black_seq_count = 0
                while row_i >= 0 and col_j < len(self.grid[0]):
                    coin = self.grid[row_i][col_j]
                    if coin == 'R':
                        red_seq_count += 1
                        black_seq_count = 0
                    elif coin == 'B':
                        red_seq_count = 0
                        black_seq_count += 1
                    else:
                        red_seq_count = 0
                        black_seq_count = 0

                    if red_seq_count == 4:
                        return float("inf")
                    elif black_seq_count == 4:
                        return float("-inf")
                    row_i -= 1
                    col_j += 1

            return None

        if (cr := check_rows()) is not None:
            return cr
        if (cc := check_cols()) is not None:
            return cc
        if (cd := check_diagonals()) is not None:
            return cd
        if self.is_board_full():
            return 0

        return None


class Agent(object):
    """Abstract class, extended by classes RandomAgent, FirstMoveAgent, MinimaxAgent.
    Do not make an instance of this class."""

    def __init__(self, color):
        """Agents use either RED or BLACK chips."""
        self.color = color

    @abc.abstractmethod
    def move(self, game):
        """Abstract. Must be implemented by a class that extends Agent."""
        pass


class RandomAgent(Agent):
    """Naive agent -- always performs a random move"""

    def move(self, game):
        """Returns a random move"""
        # YOU FILL THIS IN
        poss_moves = game.possible_moves()
        # random.seed()
        move = random.choice(poss_moves)

        return move


class FirstMoveAgent(Agent):
    """Naive agent -- always performs the first move"""

    def move(self, game):
        """Returns the first possible move"""
        # YOU FILL THIS IN
        poss_moves = game.possible_moves()
        # game.order_moves(poss_moves)
        move = poss_moves[0]

        return move


class MinimaxAgent(Agent):
    """Smart agent -- uses minimax to determine the best move"""

    def __init__(self, color):
        """Agents use either RED or BLACK chips."""
        super(MinimaxAgent, self).__init__(color)
        self.visited_games = {}  # transposition table of previously visited game positions.

    class Node():
        """The Node class is used to create a minimax tree."""

        def __init__(self, game, node_color='R', alpha=float('-inf'), beta=float('inf')):
            self.game = game
            self.node_color = node_color
            if node_color == 'R':
                self.utility = float("-inf")  # utility value of the node.
            else:
                self.utility = float("inf")
            self.best_move = -1  # the best move the node can make.
            self.alpha = alpha
            self.beta = beta

    def minimax(self, node, depth):
        """
        Returns the probable best move by using the minimax algorithm with alpha-beta pruning.
        The depth variable determines the depth limit of the search tree. Root is at 0.
        """
        #node.utility = node.game.utility()
        if (ws := node.game.winning_state()) is not None:  # reached terminal state.
            node.utility = ws
            return node

        if depth == 0:  # base case.
            # Avoid recomputing utilities of previously visited games.
            # Check if the utility of the game is already stored in the transposition table.
            if (str_grid := str(node.game.grid)) in self.visited_games:
                node.utility = self.visited_games[str_grid]
            elif (str_mir_grid := str(node.game.get_grid_mirror_symmetry())) in self.visited_games:
                node.utility = self.visited_games[str_mir_grid]
            else:
                node.utility = node.game.utility()
                self.visited_games[str_grid] = node.utility
            return node

        possible_moves = node.game.possible_moves()
        node.game.order_moves(possible_moves)

        if node.node_color == 'R':  # max node.
            for move in possible_moves:
                child_node = MinimaxAgent.Node(node.game.neighbor(move, 'R'), 'B', node.alpha, node.beta)
                child_utility = self.minimax(child_node, depth - 1).utility
                if child_utility > node.utility:
                    node.utility = child_utility
                    node.best_move = move
                node.alpha = max(node.alpha, node.utility)

                # Alpha-beta pruning.
                if node.alpha >= node.beta:
                    break
        else:                    # min node.
            for move in possible_moves:
                child_node = MinimaxAgent.Node(node.game.neighbor(move, 'B'), 'R', node.alpha, node.beta)
                child_utility = self.minimax(child_node, depth - 1).utility
                if child_utility < node.utility:
                    node.utility = child_utility
                    node.best_move = move
                node.beta = min(node.beta, node.utility)

                # Alpha-beta pruning.
                if node.alpha >= node.beta:
                    break

        return node

    def move(self, game):
        """Returns the best move using minimax"""
        # YOU FILL THIS IN
        root = MinimaxAgent.Node(game, self.color)
        depth = 5
        agent_move = self.minimax(root, depth).best_move
        #print(f"move: {agent_move}")
        return agent_move


def tournament(simulations=50):
    """Simulate connect four games, of a minimax agent playing against a random agent"""

    redwin, blackwin, tie = 0,0,0
    print()
    for i in range(simulations):
        print(i, end=" ")
        game = single_game(io=False)

        if game.winning_state() == float("inf"):
            redwin += 1
        elif game.winning_state() == float("-inf"):
            blackwin += 1
        elif game.winning_state() == 0:
            tie += 1

    print("Red %d (%.0f%%) Black %d (%.0f%%) Tie %d" % (redwin,redwin/simulations*100,blackwin,blackwin/simulations*100,tie))

    return redwin/simulations


def single_game(io=True):
    """Create a game and have two agents play it."""

    game = Game([['-' for i in range(8)] for j in range(8)])   # 8x8 empty board
    if io:
        game.display()

    maxplayer = MinimaxAgent('R')
    minplayer = RandomAgent('B')

    while True:

        m = maxplayer.move(game)
        game = game.neighbor(m, maxplayer.color)
        if io:
            time.sleep(1)
            game.display()

        if game.winning_state() is not None:
            break

        m2 = minplayer.move(game)
        game = game.neighbor(m2, minplayer.color)
        if io:
            time.sleep(1)
            game.display()

        if game.winning_state() is not None:
            break

    if game.winning_state() == float("inf"):
        print("RED WINS!")
    elif game.winning_state() == float("-inf"):
        print("BLACK WINS!")
        #print("move made by red: " + str(m))
        #game.display()
    elif game.winning_state() == 0:
        print("TIE!")

    return game


if __name__ == '__main__':
    single_game(io=True)
    #tournament(simulations=50)