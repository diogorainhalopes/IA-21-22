# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo al015:
# 95578 Francisco Ribeiro
# 96732 Diogo Lopes

import sys
import copy
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search
from utils import manhattan_distance


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        self.actions_taken = []
        self.zeros = []
        self.total_zeros = 0
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    def __init__(self, size, board):
        self.size = size
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        if (row >= self.size or col >= self.size):
            return None
        if (row < 0 or col < 0):
            return None
        return self.board[row][col]
    
    def set_number(self, row: int, col: int, val: int):
        """ Substitui o valor na respetiva posição do tabuleiro. """
        if (row >= self.size or col >= self.size):
            return
        if (row < 0 or col < 0):
            return
        self.board[row][col] = val
    
    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente abaixo e acima, 
        respectivamente. """
        return (self.get_number(row+1, col), self.get_number(row-1, col))
    
    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente à esquerda e à direita, 
        respectivamente. """
        return (self.get_number(row, col-1), self.get_number(row, col+1))

    def test_adj(self, val: int, vert: tuple, horz: tuple) -> bool:
        """ Verifica se val se encontra nas posicoes adjacentes """
        return (val in vert or val in horz)

    def no_missing_adj(self, row: int, col: int) -> bool:
        """ True se a posicao tiver todos os adjacentes """
        val = self.get_number(row, col)
        vert = self.adjacent_vertical_numbers(row, col)
        horz = self.adjacent_horizontal_numbers(row, col)
        if val == None:
            return True
        if val == 1:
            return self.test_adj(val+1, vert, horz)
        if val == self.size**2:
            return self.test_adj(val-1, vert, horz)
        return self.test_adj(val-1, vert, horz) and self.test_adj(val+1, vert, horz)
    
    def nr_zero_adj(self, row: int, col: int) -> bool:
        """ Devolve o número de 0s adjacentes """
        left = self.adjacent_horizontal_numbers(row, col)[0]
        right = self.adjacent_horizontal_numbers(row, col)[1]
        up = self.adjacent_vertical_numbers(row, col)[1]
        down = self.adjacent_vertical_numbers(row, col)[0]
        ret = 0
        if left == 0:
            ret += 1
        if right == 0:
            ret += 1
        if up == 0:
            ret += 1
        if down == 0:
            ret += 1
        return ret

    @staticmethod    
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        f = open(filename, 'r')
        size = int(f.readline())
        
        board = [[None for _ in range(size)] for _ in range(size)]
        i=0 # Linha da Board
        j=0 # Coluna da Board
        for x in f:
            k=0
            while x[k] != "\n":
                if x[k] != "\t":
                    value = x[k]
                    l = 1
                    while x[k+l] != "\t" and x[k+l] != "\n":
                        value += x[k+l]
                        l += 1
                    if l > 1:
                        k = k+l-1
                    board[i][j] = int(value)
                    j += 1
                k += 1
            i += 1
            j = 0

        return Board(size, board)
                
    def to_string(self) -> str:
        """ Imprime o tabuleiro """
        ret = ""
        i = 0
        j = 0

        for _ in range(self.size * self.size):
            if j != 0:
                ret += "\t"
            ret += str(Board.get_number(self, i, j))
            j += 1
            if j == self.size:
                i += 1
                j = 0
                ret += "\n"

        return ret[:-1]


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)
        self.min_start_val = board.size**2
        self.max_start_val = 0

    def actions(self, state: NumbrixState) -> list:
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """

        def sort_actions(lst: list, pos: tuple, size: int):
            min_man = manhattan_distance(pos, (0, 0))
            corner = (0, 0)
            if manhattan_distance(pos, (size-1, 0)) < min_man:
                min_man = manhattan_distance(pos, (size-1, 0))
                corner = (size-1, 0)
            if manhattan_distance(pos, (0, size-1)) < min_man:
                min_man = manhattan_distance(pos, (0, size-1))
                corner = (0, size-1)
            if manhattan_distance(pos, (size-1, size-1)) < min_man:
                min_man = manhattan_distance(pos, (size-1, size-1))
                corner = (size-1, size-1)

            lst.sort(key = lambda p: manhattan_distance(p, corner))
            lst.sort(key=lambda p: p[2], reverse=True)
            return lst

        ret = []
        num_dict = {}
        for i in range(state.board.size):
            for j in range(state.board.size):
                reference = state.board.get_number(i, j)
                if reference != 0:
                    num_dict[reference] = (i, j)
                    if state.id == self.initial.id:
                        if reference < self.min_start_val:
                            self.min_start_val = reference
                        elif reference > self.max_start_val:
                            self.max_start_val = reference
        
        keys = sorted(num_dict.keys())

        for key in keys:
            i = num_dict[key][0]
            j = num_dict[key][1]

            reference = state.board.get_number(i, j)

            left = state.board.adjacent_horizontal_numbers(i, j)[0]
            right = state.board.adjacent_horizontal_numbers(i, j)[1]
            up = state.board.adjacent_vertical_numbers(i, j)[1]
            down = state.board.adjacent_vertical_numbers(i, j)[0]
            
            if left == 0:
                if (reference+1 not in num_dict) and ((i, j-1, reference+1) not in state.actions_taken) and (reference < state.board.size**2):
                    ret += [(i, j-1, reference+1)]
                elif (reference-1 not in num_dict) and ((i, j-1, reference-1) not in state.actions_taken) and (reference > 1):
                    ret += [(i, j-1, reference-1)]
                    
            if right == 0:
                if (reference+1 not in num_dict) and ((i, j+1, reference+1) not in state.actions_taken) and (reference < state.board.size**2):
                    ret += [(i, j+1, reference+1)]
                elif (reference-1 not in num_dict) and ((i, j+1, reference-1) not in state.actions_taken) and (reference > 1):
                    ret += [(i, j+1, reference-1)]
            
            if up == 0:
                if (reference+1 not in num_dict) and ((i-1, j, reference+1) not in state.actions_taken) and (reference < state.board.size**2):
                    ret += [(i-1, j, reference+1)]
                elif (reference-1 not in num_dict) and ((i-1, j, reference-1) not in state.actions_taken) and (reference > 1):
                    ret += [(i-1, j, reference-1)]
            
            if down == 0:
                if (reference+1 not in num_dict) and ((i+1, j, reference+1) not in state.actions_taken) and (reference < state.board.size**2):
                    ret += [(i+1, j, reference+1)]
                elif (reference-1 not in num_dict) and ((i+1, j, reference-1) not in state.actions_taken) and (reference > 1):
                    ret += [(i+1, j, reference-1)]
            

            ret.sort(key=lambda x: x[2], reverse=True)
            if (len(ret)):
                if ret[0][2] < self.min_start_val:
                    for k in range(ret[0][2]+1, state.board.size**2+1):
                        if k not in keys:
                            ret = []
                            break
                    if len(ret) == 0:
                        continue
                break

        n_ret = sort_actions(ret, (i, j), state.board.size)
        
        return n_ret

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de 
        self.actions(state). """
        if action not in self.actions(state):
            return None
        new_board = copy.deepcopy(state.board)
        new_board.set_number(action[0], action[1], action[2])
        state.actions_taken += [action]
        return NumbrixState(new_board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. """
        for i in range(state.board.size):
            for j in range(state.board.size):
                val = state.board.get_number(i, j)
                vert = state.board.adjacent_vertical_numbers(i, j)
                horz = state.board.adjacent_horizontal_numbers(i, j)
                if(not(state.board.test_adj(val+1, vert, horz)) and val < state.board.size**2):
                    return False
        return True
    

    def check_bubbles(self, state: NumbrixState, num_dict: dict):
        """ Função de cálculo de bolhas, retorna True se pelo menos uma bolha existir """
        open_vals = []
        closed = []
        dist_check = {}
        open_vals.append(state.zeros[0])
        while len(open_vals):
            locked_adj = 0
            z = open_vals.pop(0)
            if z in closed:
                continue
            if z in state.zeros:
                state.zeros.remove(z)
            
            left = state.board.adjacent_horizontal_numbers(z[0], z[1])[0]
            right = state.board.adjacent_horizontal_numbers(z[0], z[1])[1]
            up = state.board.adjacent_vertical_numbers(z[0], z[1])[1]
            down = state.board.adjacent_vertical_numbers(z[0], z[1])[0]

            if (z[0], z[1]-1) not in closed and left != None:
                if left == 0:
                    if (z[0], z[1]-1) not in open_vals:
                        open_vals.append((z[0], z[1]-1))
                        if (z[0], z[1]-1) in state.zeros:
                            state.zeros.remove((z[0], z[1]-1))
                elif not (state.board.no_missing_adj(z[0], z[1]-1)):
                    if left not in dist_check:
                        dist_check[left] = (z[0], z[1]-1)
                else:
                    locked_adj += 1
            else:
                locked_adj += 1

            if (z[0], z[1]+1) not in closed and right != None:
                if right == 0:
                    if (z[0], z[1]+1) not in open_vals:
                        open_vals.append((z[0], z[1]+1))
                        if (z[0], z[1]+1) in state.zeros:
                            state.zeros.remove((z[0], z[1]+1))
                elif not (state.board.no_missing_adj(z[0], z[1]+1)):
                    if right not in dist_check:
                        dist_check[right] = (z[0], z[1]+1)
                else:
                    locked_adj += 1
            else:
                locked_adj += 1

            if (z[0]-1, z[1]) not in closed and up != None:
                if up == 0:
                    if (z[0]-1, z[1]) not in open_vals:
                        open_vals.append((z[0]-1, z[1]))
                        if (z[0]-1, z[1]) in state.zeros:
                            state.zeros.remove((z[0]-1, z[1]))
                elif not (state.board.no_missing_adj(z[0]-1, z[1])):
                    if up not in dist_check:
                        dist_check[up] = (z[0]-1, z[1])
                else:
                    locked_adj += 1
            else:
                locked_adj += 1

            if (z[0]+1, z[1]) not in closed and down != None:
                if down == 0:
                    if (z[0]+1, z[1]) not in open_vals:
                        open_vals.append((z[0]+1, z[1]))
                        if (z[0]+1, z[1]) in state.zeros:
                            state.zeros.remove((z[0]+1, z[1]))
                elif not (state.board.no_missing_adj(z[0]+1, z[1])):
                    if down not in dist_check:
                        dist_check[down] = (z[0]+1, z[1])
                else:
                    locked_adj += 1
            else:
                locked_adj += 1
            
            closed.append(z)
            
            if locked_adj == 4 and len(open_vals) == 0 and len(dist_check.keys()) == 0:
                return True
            
            if locked_adj == 4 and state.board.nr_zero_adj(z[0], z[1]) == 1:
                if manhattan_distance(z, num_dict[self.min_start_val]) > self.min_start_val-1 and manhattan_distance(z, num_dict[self.max_start_val]) > state.board.size**2-self.max_start_val:
                    return True


        if len(dist_check.keys()) == 2:
            key1 = list(dist_check.keys())[0]
            key2 = list(dist_check.keys())[1]

            if abs(key1-key2)-1 > len(closed) and abs(key1-key2)-1 > state.total_zeros - len(closed) and key1 != self.min_start_val and key2 != self.min_start_val:
                for k in range(min(key1, key2)+1, max(key1, key2)):
                    if k in num_dict:
                        return False
                return True

        elif len(dist_check.keys()) == 1:
            key = list(dist_check.keys())[0]
            if key-1 != len(closed) and state.board.size**2-key != len(closed):
                return True

        return False
            

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        max_val = 0
        num_dict = {}
        node.state.zeros = []
        dists = 0
        for i in range(node.state.board.size):
            for j in range(node.state.board.size):
                reference = node.state.board.get_number(i, j)
                if reference != 0:
                    num_dict[reference] = (i, j)
                    if (reference > max_val):
                        max_val = reference
                else:
                    node.state.zeros.append((i, j))
                    node.state.total_zeros += 1

        for key in num_dict:          
            if node.state.board.nr_zero_adj(num_dict[key][0], num_dict[key][1]) == 0:
                if key-1 not in num_dict and key != 1:
                    # Não tem solução
                    return node.state.board.size**3
                
                if key+1 not in num_dict and key != node.state.board.size**2:
                    # Não tem solução
                    return node.state.board.size**3
        
            elif node.state.board.nr_zero_adj(num_dict[key][0], num_dict[key][1]) == 1:
                if key-1 not in num_dict and key+1 not in num_dict and key != 1 and key != node.state.board.size**2:
                    # Não tem solução
                    return node.state.board.size**3


            if key+1 in num_dict:
                if manhattan_distance(num_dict[key], num_dict[key+1]) > 1:
                    # Não tem solução
                    return node.state.board.size**3
            elif key != max_val:
                k=2
                while key+k not in num_dict and key+k <= max_val:
                    k += 1
                if key+k <= max_val and manhattan_distance(num_dict[key], num_dict[key+k]) > k:
                    # Não tem solução
                    return node.state.board.size**3
                dists += abs(k-manhattan_distance(num_dict[key], num_dict[key+k]))
                # Outra heurística testada: manhattan_distance(num_dict[key], num_dict[key+k])-1)

        while len(node.state.zeros) > 0:
            if self.check_bubbles(node.state, num_dict):
                # Não tem solução (bolha)
                return node.state.board.size**3
    
        return dists


if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    goal_node = greedy_search(problem)
    print(goal_node.state.board.to_string())