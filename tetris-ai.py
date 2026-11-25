"""Tetris AI: a Tetris-style game that teaches itself to play with deep Q-learning.

Run `python tetris-ai.py`. Play with the arrow keys, or press "Auto Learning"
to let the neural network play and train on many games in parallel.
"""
import os
import sys
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
COLORS          = ['gray', 'lightgreen', 'pink', 'blue', 'orange', 'purple', 'red', 'yellow', 'white']
FIELD_HEIGHT    = 20      # board rows
FIELD_WIDTH     = 10      # board columns
PIECE_SIZE      = 30      # pixel size of one board cell
BLOCK_SIZE      = 20      # pixel size of one cell in the "next piece" preview
SCORE_PER_ELIMINATED_LINES = (0, 100, 500, 2500, 10000)  # indexed by lines cleared at once
PARALLEL_TASK_COUNT = 50  # games trained at once (1 shown on screen + hidden workers)
CONSOLE_RENDER = True     # also draw the board in the terminal with ANSI colors
WEIGHT_DIR = './weight'   # where trained networks (.wgt) are saved and loaded

# Each piece is a list of (row, col) cells relative to its offset.
TETROMINOS = [
        [(0, 0), (0, 1), (1, 0), (1, 1)], # O
        [(0, 0), (0, 1), (1, 1), (2, 1)], # L
        [(0, 1), (1, 1), (2, 1), (2, 0)], # J
        [(0, 1), (1, 0), (1, 1), (2, 0)], # Z
        [(0, 1), (1, 0), (1, 1), (2, 1)], # T
        [(0, 0), (1, 0), (1, 1), (2, 1)], # S
        [(0, 1), (1, 1), (2, 1), (3, 1)]  # I
]


# ---------------------------------------------------------------------------
# Reinforcement learning
# ---------------------------------------------------------------------------
class GameNet(nn.Module):
    """Q-network: maps the 20x10 board to a Q-value for each of the 5 actions
    (none, left, right, rotate, down)."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(FIELD_HEIGHT * FIELD_WIDTH, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 5)

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32)
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def saveGameNet(self, file_name=None):
        """Save weights to WEIGHT_DIR (timestamped name by default). Returns the path."""
        os.makedirs(WEIGHT_DIR, exist_ok=True)

        if file_name is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'GameQNet_{ts}.wgt'

        if not os.path.isabs(file_name):
            file_name = os.path.join(WEIGHT_DIR, file_name)

        torch.save(self.state_dict(), file_name)
        return file_name

    def loadGameNet(self, file_name=None):
        """Load weights from file_name, or the newest .wgt in WEIGHT_DIR.
        Returns the loaded path, or None on failure."""
        if file_name is None:
            if not os.path.exists(WEIGHT_DIR):
                print("Error: Unable to load state dictionary: weight folder not found")
                return None

            candidates = [
                os.path.join(WEIGHT_DIR, f)
                for f in os.listdir(WEIGHT_DIR)
                if f.endswith('.wgt')
            ]
            if not candidates:
                print("Error: No model files found in weight folder")
                return None
            file_name = max(candidates, key=os.path.getmtime)
        elif not os.path.isabs(file_name):
            file_name = os.path.join(WEIGHT_DIR, file_name)

        try:
            self.load_state_dict(torch.load(file_name, map_location='cpu'))
            self.eval()
            return file_name
        except Exception:
            print("Error: Unable to load state dictionary: {}".format(file_name))
            return None


class GameTrainer:
    """Updates the Q-network one transition at a time (online Q-learning)."""

    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma  # discount factor for future rewards
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def trainStep(self, state, action_idx, reward, next_state, done):
        """Move Q(state, action) toward reward + gamma * max Q(next_state). Returns the loss."""
        self.model.train()
        self.optimizer.zero_grad()
        pred_q = self.model(state)
        target_q = pred_q.detach().clone()

        with torch.no_grad():
            if done:
                q_new = float(reward)
            else:
                q_next = torch.max(self.model(next_state)).item()
                q_new = float(reward) + self.gamma * q_next

        # Only the taken action's target changes; other actions contribute zero loss.
        target_q[action_idx] = q_new
        loss = self.criterion(pred_q, target_q)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.model.eval()
        return loss.item()


class Agent:
    """Chooses actions (epsilon-greedy) and scores board states (reward shaping)."""

    def __init__(self):
        self.totalGames = 0
        self.epsilon = 1.0
        self.epsilonMin = 0.05
        self.epsilonDecayGames = 220      # games until epsilon decays to epsilonMin
        self.openingRandomSteps = 60      # extra exploration for the first N steps of each game
        self.openingEpsilonBoost = 0.20
        # Random-action probabilities. Action order: None, Left, Right, Rotate, Down
        self.randomActionWeights = [0.03, 0.24, 0.24, 0.27, 0.22]
        self.downActionBias = 0.08        # nudges the greedy policy toward dropping pieces
        self.currentGameSteps = 0

    def onNewGame(self):
        self.currentGameSteps = 0

    def createMapState(self, statelist):
        """Turn the flat list of cell colors into a (FIELD_HEIGHT, FIELD_WIDTH) array."""
        return np.array(statelist).reshape(FIELD_HEIGHT, FIELD_WIDTH)

    def createRewardFactors(self, state):
        """Per-column/row board features used by getReward.

        Returns (first, bump, height, piece, hole, line):
          first  - row index of the top filled cell per column (-1 if empty)
          bump   - span from top to bottom filled cell per column
          height - column height
          piece  - filled cells per column
          hole   - empty cells under the column's top cell
          line   - filled cells per row
        """
        checked_1st_nonzero = self.first_nonzero(state, axis=0, invalid_val=-1)
        checked_last_nonzero = self.last_nonzero(state, axis=0, invalid_val=FIELD_HEIGHT)

        first_list = checked_1st_nonzero

        bump_list = checked_last_nonzero - checked_1st_nonzero + 1
        bump_list[bump_list > FIELD_HEIGHT] = 0

        height_list = FIELD_HEIGHT - checked_1st_nonzero
        height_list[height_list > FIELD_HEIGHT] = 0  # empty columns

        piece_list = np.count_nonzero(state, axis=0)
        hole_list = height_list - piece_list
        line_list = np.count_nonzero(state, axis=1)

        return first_list, bump_list, height_list, piece_list, hole_list, line_list

    # axis==0: check columns and write output into a row
    # axis==1: check rows and write output into a column
    def first_nonzero(self, state, axis, invalid_val=-1):
        mask = state != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def last_nonzero(self, state, axis, invalid_val=FIELD_HEIGHT):
        mask = state != 0
        val = state.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def getAction(self, model, state):
        """Epsilon-greedy: random weighted action while exploring, else the network's best."""
        base_epsilon = max(self.epsilonMin, 1.0 - (self.totalGames / self.epsilonDecayGames))
        opening_boost = self.openingEpsilonBoost if self.currentGameSteps < self.openingRandomSteps else 0.0
        self.epsilon = min(1.0, base_epsilon + opening_boost)

        if random.random() < self.epsilon:
            move = random.choices(range(5), weights=self.randomActionWeights, k=1)[0]
        else:
            predAction = model(state).detach().clone()
            predAction[4] += self.downActionBias
            move = torch.argmax(predAction).item()

        self.currentGameSteps += 1
        return move

    def getReward(self, myRewardFactors, myEliminated, state):
        """Hand-tuned reward: big bonus for line clears, penalties for tall,
        bumpy, holey boards, bonuses for compact rows near the floor."""
        _, _, height_rlist, _, hole_rlist, line_clist = myRewardFactors

        # Rows that are close to full score higher.
        lcl = float(np.sum(100.0 / (FIELD_WIDTH - line_clist + 1.0)))

        aggregate_height = float(np.sum(height_rlist))
        max_height = float(np.max(height_rlist))
        bumpiness = float(np.sum(np.abs(np.diff(height_rlist))))
        holes = float(np.sum(hole_rlist))
        danger_cols = float(np.sum(height_rlist >= (FIELD_HEIGHT - 4)))

        board_penalty = (
            -4.0 * aggregate_height
            -16.0 * holes
            -7.0 * bumpiness
            -12.0 * max(0.0, max_height - 12.0) ** 2
            -220.0 * danger_cols
        )

        # Reward side-by-side blocks, punish blocks with no horizontal neighbor.
        occupied = (state != 0)
        horizontal_pairs = float(np.sum(occupied[:, :-1] & occupied[:, 1:]))
        compact_bonus = 24.0 * horizontal_pairs

        has_left_neighbor = np.zeros_like(occupied, dtype=bool)
        has_right_neighbor = np.zeros_like(occupied, dtype=bool)
        has_left_neighbor[:, 1:] = occupied[:, :-1]
        has_right_neighbor[:, :-1] = occupied[:, 1:]
        isolated_blocks = float(np.sum(occupied & ~(has_left_neighbor | has_right_neighbor)))
        separation_penalty = -30.0 * isolated_blocks

        # Reward placements near the floor much more than near the top.
        row_fill_ratio = np.sum(occupied, axis=1) / float(FIELD_WIDTH)
        bottom_weights = (np.arange(FIELD_HEIGHT, dtype=np.float32) + 1.0) / float(FIELD_HEIGHT)
        bottom_fill_bonus = 120.0 * float(np.dot(row_fill_ratio, bottom_weights ** 2))

        line_clear_reward = [0.0, 1200.0, 5000.0, 15000.0, 50000.0]
        ell = line_clear_reward[min(myEliminated, 4)] + (myEliminated ** 2) * 1200.0

        return lcl + board_penalty + compact_bonus + separation_penalty + bottom_fill_bonus + ell


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------
class Tetris:
    """Board state and rules. field[r][c] holds a color id (0 = empty)."""

    def __init__(self):
        self.field = [[0 for c in range(FIELD_WIDTH)] for r in range(FIELD_HEIGHT)]
        self.score = 0
        self.level = 0
        self.total_lines_eliminated = 0
        self.lines_eliminated = 0  # lines cleared by the last placed piece
        self.gameOver = False
        self.next_tetromino_id = random.randint(1, len(TETROMINOS))
        self.reset_tetromino()

    def reset_tetromino(self):
        """Spawn the next piece above the board; game over if it has no room."""
        self.tetromino_id = self.next_tetromino_id
        self.next_tetromino_id = random.randint(1, len(TETROMINOS))
        self.tetromino = TETROMINOS[self.tetromino_id-1]
        self.tetromino_color = self.tetromino_id
        self.tetromino_offset = [-2, FIELD_WIDTH//2]
        self.gameOver = not any(self.is_cell_free(r, c) for (r, c) in self.get_tetromino_coords())

    def get_tetromino_coords(self):
        return [(r+self.tetromino_offset[0], c + self.tetromino_offset[1]) for (r, c) in self.tetromino]

    def apply_tetromino(self):
        """Lock the piece, clear full rows, update score/level, spawn the next piece."""
        for (r, c) in self.get_tetromino_coords():
            self.field[r][c] = self.tetromino_color

        new_field = [row for row in self.field if any(tile == 0 for tile in row)]
        self.lines_eliminated = len(self.field)-len(new_field)
        self.total_lines_eliminated += self.lines_eliminated
        self.field = [[0]*FIELD_WIDTH for x in range(self.lines_eliminated)] + new_field
        self.score += SCORE_PER_ELIMINATED_LINES[self.lines_eliminated] * (self.level + 1)
        self.level = self.total_lines_eliminated // 10
        self.reset_tetromino()

    def get_color(self, r, c):
        """Color at (r, c), including the falling piece."""
        return self.tetromino_color if (r, c) in self.get_tetromino_coords() else self.field[r][c]

    def is_cell_free(self, r, c):
        # Rows above the board (r < 0) count as free so pieces can spawn there.
        return r < FIELD_HEIGHT and 0 <= c < FIELD_WIDTH and (r < 0 or self.field[r][c] == 0)

    def move(self, dr, dc):
        """Shift the piece. A blocked downward move locks it in place. Returns True if it moved."""
        if self.gameOver:
            return False

        if all(self.is_cell_free(r + dr, c + dc) for (r, c) in self.get_tetromino_coords()):
            self.tetromino_offset = [self.tetromino_offset[0] + dr, self.tetromino_offset[1] + dc]
            return True
        elif dr == 1 and dc == 0:
            self.gameOver = any(r < 0 for (r, c) in self.get_tetromino_coords())
            if not self.gameOver:
                self.apply_tetromino()
            return False

        return False

    def rotate(self):
        """Rotate the piece 90 degrees, trying a few small offsets if it collides."""
        if self.gameOver:
            return False

        ys = [r for (r, c) in self.tetromino]
        xs = [c for (r, c) in self.tetromino]
        size = max(max(ys) - min(ys), max(xs) - min(xs))
        rotated_tetromino = [(c, size-r) for (r, c) in self.tetromino]

        # Constrained wall kicks: avoid large automatic jumps that make pieces feel erratic.
        kick_tests = [
            (0, 0),
            (0, -1),
            (0, 1),
            (-1, 0),
            (-1, -1),
            (-1, 1),
            (0, -2),
            (0, 2),
            (-2, 0),
        ]

        for kick_r, kick_c in kick_tests:
            test_offset = [self.tetromino_offset[0] + kick_r, self.tetromino_offset[1] + kick_c]
            tetromino_coord = [(r + test_offset[0], c + test_offset[1]) for (r, c) in rotated_tetromino]
            if all(self.is_cell_free(r, c) for (r, c) in tetromino_coord):
                self.tetromino, self.tetromino_offset = rotated_tetromino, test_offset
                return True

        return False

    def restart(self):
        if self.gameOver:
            self.__init__()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class GameApp(tk.Frame):
    """Tkinter window: draws the board, handles keys/buttons, and drives training."""

    def __init__(self, master=None):
        super().__init__(master)
        self.gamma = 0.9   # discount rate for future rewards
        self.lr = 0.001    # learning rate
        self.autoLearning = False
        self.tetris = Tetris()
        self.agent = Agent()
        self.agent.onNewGame()
        self.model = GameNet()
        self.trainer = GameTrainer(self.model, self.lr, self.gamma)
        # Hidden games that train the same network alongside the visible one.
        self.parallelWorkers = []
        for _ in range(max(0, PARALLEL_TASK_COUNT - 1)):
            worker_agent = Agent()
            worker_agent.onNewGame()
            self.parallelWorkers.append({"tetris": Tetris(), "agent": worker_agent})
        self.latestLoss = 0.0
        self.consoleRenderTick = 0
        self.consoleRenderEvery = 8  # draw the terminal board every N frames
        self.mapState = None
        self.create_widgets()
        self.pack()
        self.update_clock()

    def update_clock(self):
        """Game tick: gravity, redraw, schedule the next tick (fast while training)."""
        self.tetris.move(1, 0)
        self.update()
        self.master.after(1 if self.autoLearning else 660, self.update_clock)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, height=PIECE_SIZE*FIELD_HEIGHT,
                                      width=PIECE_SIZE*FIELD_WIDTH, bg="black", bd=0)
        self.canvas.bind('<Left>',   lambda _: (self.tetris.move(0, -1), self.update()))
        self.canvas.bind('<Right>',  lambda _: (self.tetris.move(0, 1), self.update()))
        self.canvas.bind('<Down>',   lambda _: (self.tetris.move(1, 0), self.update()))
        self.canvas.bind('<Up>',     lambda _: (self.tetris.rotate(), self.update()))
        self.canvas.focus_set()
        self.rectangles = [
            self.canvas.create_rectangle(c*PIECE_SIZE, r*PIECE_SIZE, (c+1)*PIECE_SIZE, (r+1)*PIECE_SIZE)
                for r in range(FIELD_HEIGHT) for c in range(FIELD_WIDTH)
            ]
        self.canvas.pack(side="left")

        self.status_msg = tk.Label(self, anchor='w', width=20, font=("Courier", 24, 'bold'))
        self.status_msg.pack(side="top")

        # "Next piece" preview (4 rows x 2 cols).
        self.block = tk.Canvas(self, height=BLOCK_SIZE*4, width=BLOCK_SIZE*2, bd=0)
        self.brect = [
            self.block.create_rectangle(c*BLOCK_SIZE, r*BLOCK_SIZE, (c+1)*BLOCK_SIZE, (r+1)*BLOCK_SIZE)
                for r in range(4) for c in range(2)
           ]
        self.block.pack(side="top")

        # Text dump of the board's color ids (what the network sees).
        self.mapState_msg = tk.Label(self, anchor='center', width=11, font=("Courier", 12, 'bold'), fg='white', bg='black')
        self.mapState_msg.pack(side="top")

        self.saveButton = tk.Button(self, text='Save Network', anchor='center', width=12,
                        activeforeground="red", command=self.saveModel)
        self.saveButton.pack(side='top')

        self.loadButton = tk.Button(self, text='Load Network', anchor='center', width=12,
                        activeforeground="red", command=self.loadModel)
        self.loadButton.pack(side='top')

        self.autoLearnButton = tk.Button(self, text='Auto Learning', anchor='center', width=12,
                                    activeforeground="red", state=tk.NORMAL, relief='raised', command=self.autoLearningGame)
        self.autoLearnButton.pack(side='top')

        self.manualLearnButton = tk.Button(self, text='Manual Learning', anchor='center', width=12,
                                    activeforeground="red", state=tk.ACTIVE, relief='groove', command=self.manualLearningGame)
        self.manualLearnButton.pack(side='top')

        self.restartButton = tk.Button(self, text='Restart', anchor='center', width=12,
                                    activeforeground="red", command=self.tetris.restart)
        self.restartButton.pack(side="top")

        self.exitButton = tk.Button(self, text='Exit', anchor='center', width=12,
                                    activeforeground="red", command=sys.exit)
        self.exitButton.pack(side="top")

    def autoLearningGame(self):
        """Switch to AI mode, continuing from the newest saved network if there is one."""
        if not self.autoLearning:
            self.model.loadGameNet()
            self.autoLearning = True
            self.autoLearnButton.configure(state=tk.ACTIVE, relief='groove')
            self.manualLearnButton.configure(state=tk.NORMAL, relief='raised')

    def manualLearningGame(self):
        """Switch back to human play."""
        if self.autoLearning:
            self.autoLearning = False
            self.manualLearnButton.configure(state=tk.ACTIVE, relief='groove')
            self.autoLearnButton.configure(state=tk.NORMAL, relief='raised')

    def saveModel(self):
        saved_path = self.model.saveGameNet()
        messagebox.showinfo('Save Network', f'Saved model:\n{saved_path}')

    def loadModel(self):
        model_folder = os.path.abspath(WEIGHT_DIR)
        if not os.path.exists(model_folder):
            messagebox.showwarning('Load Network', 'No weight folder found yet.')
            return

        selected_file = filedialog.askopenfilename(
            title='Select model file to load',
            initialdir=model_folder,
            filetypes=[('Weight files', '*.wgt'), ('All files', '*.*')],
        )

        if not selected_file:
            return

        loaded_path = self.model.loadGameNet(selected_file)
        if loaded_path is None:
            messagebox.showerror('Load Network', 'Failed to load selected model file.')
        else:
            messagebox.showinfo('Load Network', f'Loaded model:\n{loaded_path}')

    def _state_list(self, tetris_obj):
        return [tetris_obj.get_color(r, c) for r in range(FIELD_HEIGHT) for c in range(FIELD_WIDTH)]

    def _apply_move(self, tetris_obj, move):
        """Action ids: 0 none, 1 left, 2 right, 3 rotate, 4 down."""
        if move == 1:
            tetris_obj.move(0, -1)
        elif move == 2:
            tetris_obj.move(0, 1)
        elif move == 3:
            tetris_obj.rotate()
        elif move == 4:
            tetris_obj.move(1, 0)

    def _run_train_step(self, tetris_obj, agent_obj, map_state=None):
        """Play one action in a game, score it, train on it, and restart on game over."""
        if map_state is None:
            map_state = agent_obj.createMapState(self._state_list(tetris_obj))

        move = agent_obj.getAction(self.model, map_state)
        self._apply_move(tetris_obj, move)

        next_map_state = agent_obj.createMapState(self._state_list(tetris_obj))
        next_reward_factors = agent_obj.createRewardFactors(next_map_state)
        train_reward = agent_obj.getReward(next_reward_factors, tetris_obj.lines_eliminated, next_map_state)
        if tetris_obj.gameOver:
            train_reward -= 40000.0

        loss = self.trainer.trainStep(
            map_state,
            move,
            train_reward,
            next_map_state,
            tetris_obj.gameOver,
        )

        if tetris_obj.gameOver:
            agent_obj.totalGames += 1
            tetris_obj.restart()
            agent_obj.onNewGame()

        return loss, train_reward

    def _render_console_board(self):
        """Draw the visible board in the terminal using ANSI background colors."""
        ansi_bg = {
            0: "\033[40m",   # empty
            1: "\033[100m",  # gray
            2: "\033[102m",  # lightgreen
            3: "\033[105m",  # pink
            4: "\033[104m",  # blue
            5: "\033[43m",   # orange-ish
            6: "\033[45m",   # purple
            7: "\033[101m",  # red
            8: "\033[103m",  # yellow
        }
        reset = "\033[0m"
        dim = "\033[90m"

        lines = []
        border = "+" + ("--" * FIELD_WIDTH) + "+"
        lines.append(border)
        for r in range(FIELD_HEIGHT):
            row_tokens = ["|"]
            for c in range(FIELD_WIDTH):
                color_id = self.tetris.get_color(r, c)
                if color_id == 0:
                    row_tokens.append(f"{dim}..{reset}")
                else:
                    row_tokens.append(f"{ansi_bg.get(color_id, ansi_bg[0])}  {reset}")
            row_tokens.append("|")
            lines.append("".join(row_tokens))
        lines.append(border)

        # Clear terminal and redraw a single animated board frame.
        print("\033[H\033[2J", end="")
        print("Tetris Console View (ANSI)")
        print("Score:{}  Level:{}  Games:{}".format(self.tetris.score, self.tetris.level, self.agent.totalGames))
        print("\n".join(lines))
        sys.stdout.flush()

    def update(self):
        """Redraw everything and, in auto-learning mode, run one training step per game."""
        # Next-piece preview
        for _id in self.brect:
            self.block.itemconfig(_id, fill='white')
        for (r, c) in TETROMINOS[self.tetris.next_tetromino_id-1]:
            _id = 2 * r + c + 1  # canvas item ids start at 1
            self.block.itemconfig(_id, fill=COLORS[self.tetris.next_tetromino_id])

        # Main board
        statelist = []
        for i, _id in enumerate(self.rectangles):
            color_id = self.tetris.get_color(i//FIELD_WIDTH, i % FIELD_WIDTH)
            self.canvas.itemconfig(_id, fill=COLORS[color_id])
            statelist.append(color_id)

        rows = ["".join(str(x) for x in statelist[i:i + FIELD_WIDTH])
                for i in range(0, len(statelist), FIELD_WIDTH)]
        self.mapState_msg['text'] = "\n".join(rows) + "\n\n"

        total_games = self.agent.totalGames + sum(w["agent"].totalGames for w in self.parallelWorkers)
        self.status_msg['text'] = "Score:%05d\nLevel:%02d\nGames:%05d\nTasks:%02d\nEps:%0.2f\nLoss:%0.2f"%(
            self.tetris.score,
            self.tetris.level,
            total_games,
            PARALLEL_TASK_COUNT,
            self.agent.epsilon,
            self.latestLoss,
        )

        if CONSOLE_RENDER:
            self.consoleRenderTick += 1
            if self.consoleRenderTick % self.consoleRenderEvery == 0:
                self._render_console_board()

        self.restartButton.configure(state=tk.ACTIVE if self.tetris.gameOver else tk.DISABLED)

        self.mapState = self.agent.createMapState(statelist)

        if self.autoLearning:
            losses = []
            main_loss, _ = self._run_train_step(self.tetris, self.agent, self.mapState)
            losses.append(main_loss)

            # Hidden workers get their own gravity tick plus one action each.
            for worker in self.parallelWorkers:
                worker["tetris"].move(1, 0)
                worker_loss, _ = self._run_train_step(worker["tetris"], worker["agent"])
                losses.append(worker_loss)

            self.latestLoss = float(np.mean(losses))

        if self.tetris.gameOver and not self.autoLearning:
            self.agent.totalGames += 1
            self.tetris.restart()
            self.agent.onNewGame()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Deep Learning Tetris Game by Daniel Chen v2')
    try:
        root.iconbitmap('icon.ico')
    except tk.TclError:
        pass
    game = GameApp(master=root)

    # Center the window on screen.
    root.update_idletasks()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    pos_x = (screen_width - window_width) // 2
    pos_y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{pos_x}+{pos_y}')

    game.mainloop()
