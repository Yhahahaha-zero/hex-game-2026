import math
import torch

from hex_board import RED, BLUE, EMPTY, ROWS, COLS
from ai.encode import encode_board


class MCTSNode:

    def __init__(
        self,
        board,
        player,
        parent=None,
        move=None,
        prior=0.0
    ):
        self.board = board
        self.player = player

        self.parent = parent
        self.move = move

        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):

        return len(self.children) > 0


class MCTS:

    def __init__(
        self,
        model,
        device="cuda",
        simulations=100,
        c_puct=1.5
    ):

        self.model = model
        self.device = device

        self.simulations = simulations
        self.c_puct = c_puct

        self.model.eval()

    # ========================================================
    # 获取合法动作
    # ========================================================

    def get_legal_moves(self, board):

        moves = []

        for r in range(ROWS):

            for q in range(COLS):

                if board.grid[r][q] == EMPTY:

                    moves.append(
                        (q, r)
                    )

        return moves

    # ========================================================
    # 创建棋盘副本
    # ========================================================

    def copy_board(self, board):

        new_board = type(board)()

        for r in range(ROWS):

            for q in range(COLS):

                new_board.grid[r][q] = (
                    board.grid[r][q]
                )

        new_board.history = list(
            board.history
        )

        return new_board

    # ========================================================
    # 网络评估
    # ========================================================

    def evaluate(self, board, player):

        state = encode_board(
            board,
            player
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            policy_logits, value = self.model(
                state
            )

        policy_logits = policy_logits[0]

        value = value.item()

        legal_moves = self.get_legal_moves(
            board
        )

        if not legal_moves:

            return {}, value

        # ----------------------------------------------------
        # 只保留合法位置
        # ----------------------------------------------------

        legal_indices = []

        for q, r in legal_moves:

            index = r * COLS + q

            legal_indices.append(
                index
            )

        legal_logits = policy_logits[
            legal_indices
        ]

        probabilities = torch.softmax(
            legal_logits,
            dim=0
        )

        priors = {}

        for i, move in enumerate(
            legal_moves
        ):

            priors[move] = (
                probabilities[i].item()
            )

        return priors, value

    # ========================================================
    # 扩展节点
    # ========================================================

    def expand(self, node):

        priors, value = self.evaluate(
            node.board,
            node.player
        )

        for move, prior in priors.items():

            q, r = move

            new_board = self.copy_board(
                node.board
            )

            new_board.place(
                q,
                r,
                node.player
            )

            next_player = (
                BLUE
                if node.player == RED
                else RED
            )

            child = MCTSNode(
                board=new_board,
                player=next_player,
                parent=node,
                move=move,
                prior=prior
            )

            node.children[move] = child

        return value

    # ========================================================
    # 选择
    # ========================================================

    def select_child(self, node):

        best_score = float("-inf")

        best_child = None

        parent_visits = max(
            node.visit_count,
            1
        )

        for child in node.children.values():

            # UCB / PUCT
            exploitation = -child.value

            exploration = (
                self.c_puct
                * child.prior
                * math.sqrt(parent_visits)
                / (
                    1
                    + child.visit_count
                )
            )

            score = (
                exploitation
                + exploration
            )

            if score > best_score:

                best_score = score

                best_child = child

        return best_child

    # ========================================================
    # 判断终局
    # ========================================================

    def get_terminal_value(
        self,
        board,
        player
    ):

        opponent = (
            BLUE
            if player == RED
            else RED
        )

        if board.has_won(opponent):

            return -1.0

        if board.has_won(player):

            return 1.0

        return None

    # ========================================================
    # 单次搜索
    # ========================================================

    def simulate(self, root):

        node = root

        # ----------------------------------------------------
        # Selection
        # ----------------------------------------------------

        while node.is_expanded():

            terminal_value = (
                self.get_terminal_value(
                    node.board,
                    node.player
                )
            )

            if terminal_value is not None:

                value = terminal_value

                break

            node = self.select_child(
                node
            )

        else:

            terminal_value = (
                self.get_terminal_value(
                    node.board,
                    node.player
                )
            )

            if terminal_value is not None:

                value = terminal_value

            else:

                # --------------------------------------------
                # Expansion
                # --------------------------------------------

                value = self.expand(
                    node
                )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        current = node

        while current is not None:

            current.visit_count += 1

            current.value_sum += value

            value = -value

            current = current.parent

    # ========================================================
    # 搜索最佳落子
    # ========================================================

    def search(
        self,
        board,
        player
    ):

        root = MCTSNode(
            board=self.copy_board(
                board
            ),
            player=player
        )

        # 第一次扩展
        self.expand(root)

        if not root.children:

            raise ValueError(
                "没有合法落子。"
            )

        # ----------------------------------------------------
        # MCTS simulations
        # ----------------------------------------------------

        for _ in range(
            self.simulations
        ):

            self.simulate(
                root
            )

        # ----------------------------------------------------
        # 根据访问次数选择
        # ----------------------------------------------------

        best_move = None

        best_visits = -1

        for move, child in root.children.items():

            if (
                child.visit_count
                > best_visits
            ):

                best_visits = (
                    child.visit_count
                )

                best_move = move

        return (
            best_move,
            best_visits,
            root
        )