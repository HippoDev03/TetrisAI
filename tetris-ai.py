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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque


# Global Constant Variables
COLORS          = ['gray', 'lightgreen', 'pink', 'blue', 'orange', 'purple', 'red', 'yellow', 'white']
FIELD_HEIGHT    = 20
FIELD_WIDTH     = 10
PIECE_SIZE      = 30
BLOCK_SIZE      = 20
SCORE_PER_ELIMINATED_LINES = (0, 100, 500, 2500, 10000)
PARALLEL_TASK_COUNT = 50
CONSOLE_RENDER = True

TETROMINOS = [
        [(0, 0), (0, 1), (1, 0), (1, 1)], # O
        [(0, 0), (0, 1), (1, 1), (2, 1)], # L
        [(0, 1), (1, 1), (2, 1), (2, 0)], # J 
        [(0, 1), (1, 0), (1, 1), (2, 0)], # Z
        [(0, 1), (1, 0), (1, 1), (2, 1)], # T
        [(0, 0), (1, 0), (1, 1), (2, 1)], # S
        [(0, 1), (1, 1), (2, 1), (3, 1)]  # I
]


# Reinforcement Learning
class GameNet(nn.Module):
    def __init__(self):
        super().__init__()

        # self.convol1=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        # self.bn1 = nn.BatchNorm2d(1)
        # self.convol2=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        # self.bn2 = nn.BatchNorm2d(1)
        # self.convol3=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        # self.bn3 = nn.BatchNorm2d(1)
        # self.convol4=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        # self.bn4 = nn.BatchNorm2d(1)
        # self.convol5=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        # self.bn5 = nn.BatchNorm2d(1)
        # #self.pool1  =nn.MaxPool2d(kernel_size=2)
        # self.linear1=nn.Linear(75, 5)

        # type 2
        self.linear1 = nn.Linear(FIELD_HEIGHT * FIELD_WIDTH, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 5)

    def forward(self, state):
        # x = torch.tensor(state, dtype=torch.float).reshape(1, 1, FIELD_HEIGHT, FIELD_WIDTH)
        # x = torch.tensor(state, dtype=torch.float)
        # x = x.unsqueeze(0)

        # x = F.relu(self.bn1(self.convol1(x)))
        # x = F.relu(self.bn2(self.convol2(x)))
        # x = F.relu(self.bn3(self.convol3(x)))
        # x = F.relu(self.bn4(self.convol4(x)))
        # x = F.relu(self.bn5(self.convol5(x)))
        # x = torch.flatten(x)
        # x = self.linear1(x)
        # return x

        x = torch.as_tensor(state, dtype=torch.float32)
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


    def saveGameNet(self, file_name=None):
        model_folder_path = './weight'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        if file_name is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'GameQNet_{ts}.wgt'

        if not os.path.isabs(file_name):
            file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)
        return file_name

    def loadGameNet(self, file_name=None):
         model_folder_path = './weight'

         if file_name is None:
             if not os.path.exists(model_folder_path):
                 print("Error: Unable to load state dictionary: weight folder not found")
                 return None

             candidates = [
                 os.path.join(model_folder_path, f)
                 for f in os.listdir(model_folder_path)
                 if f.endswith('.wgt')
             ]
             if not candidates:
                 print("Error: No model files found in weight folder")
                 return None
             file_name = max(candidates, key=os.path.getmtime)
         elif not os.path.isabs(file_name):
             file_name = os.path.join(model_folder_path, file_name)

         try:
             self.load_state_dict(torch.load(file_name, map_location='cpu'))
             self.eval()
             return file_name
         except Exception:
             print("Error: Unable to load state dictionary: {}".format(file_name))
             return None


class GameTrainer:
    def __init__(self, model, lr, gamma, oreward, nreward):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.oreward = oreward
        self.nreward = nreward
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.SmoothL1Loss()

    def trainStep(self, state, action_idx, reward, next_state, done):
        # One-step Q-learning target update for the selected action.
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

        target_q[action_idx] = q_new
        lossfn = self.criterion(pred_q, target_q)
        lossfn.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.model.eval()
        return lossfn.item()


class Agent():
    def __init__(self):
        self.totalGames = 0
        self.maxMemory = 100_000  # max memory
        self.batchSize = 1_000 # max batch training size
        self.memory = deque(maxlen=self.maxMemory)
        self.epsilonMin = 0.05
        self.epsilonDecayGames = 220
        self.openingRandomSteps = 60
        self.openingEpsilonBoost = 0.20
        # Action order: None, Left, Right, Rotate, Down
        self.randomActionWeights = [0.03, 0.24, 0.24, 0.27, 0.22]
        self.downActionBias = 0.08
        self.currentGameSteps = 0

    def onNewGame(self):
        self.currentGameSteps = 0


    def createMapState(self, statelist):
        x = np.array(statelist).reshape(FIELD_HEIGHT, FIELD_WIDTH)
        return x

    def createRewardFactors(self, state):
        checked_1st_nonzero = self.first_nonzero(state, axis=0, invalid_val=-1)
        checked_last_nonzero = self.last_nonzero(state, axis=0, invalid_val=FIELD_HEIGHT)
        
        first_list = checked_1st_nonzero

        bump_list = checked_last_nonzero - checked_1st_nonzero + 1
        for i in range(len(bump_list)):
            if(bump_list[i]>FIELD_HEIGHT):
                bump_list[i]=0
        
        height_list = np.ones(FIELD_WIDTH, dtype=int)*FIELD_HEIGHT - checked_1st_nonzero
        for i in range(len(height_list)):
            if(height_list[i]>FIELD_HEIGHT):
                height_list[i]=0

        piece_list=np.ones(FIELD_WIDTH, dtype=int)
        for i in range(FIELD_WIDTH):
            piece_list[i] = np.count_nonzero(state[:,i])

        hole_list = height_list - piece_list

        line_list = np.zeros(FIELD_HEIGHT, dtype=int)
        for i in range(len(line_list)):
            line_list[i] = np.count_nonzero(state[i,:])

        return first_list, bump_list, height_list, piece_list, hole_list, line_list

    
    # axis==0: check columns and write output into a row
    # axis==1: check rows and write output into a column 
    def first_nonzero(self, state, axis, invalid_val=-1):
        mask = state!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def last_nonzero(self, state, axis, invalid_val=FIELD_HEIGHT):
        mask = state!=0
        val = state.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def getAction(self, model, state):
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
        first_rlist, bump_rlist, height_rlist, piece_rlist, hole_rlist, line_clist = myRewardFactors

        lcl = 0.0
        for i in range(len(line_clist)):
            lcl += 100.0 / (float(FIELD_WIDTH - line_clist[i]) + 1.0)

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

        reward  = lcl + board_penalty + compact_bonus + separation_penalty + bottom_fill_bonus + ell
        return reward

        
class Tetris():
    def __init__(self):
        self.field = [[0 for c in range(FIELD_WIDTH)] for r in range(FIELD_HEIGHT)]
        self.score = 0
        self.level = 0
        self.total_lines_eliminated = 0
        self.lines_eliminated = 0
        self.gameOver = False
        self.next_tetromino_id = random.randint(1,len(TETROMINOS))
        self.reset_tetromino()

    def reset_tetromino(self):
        self.tetromino_id = self.next_tetromino_id
        self.next_tetromino_id = random.randint(1,len(TETROMINOS))
        self.tetromino = TETROMINOS[self.tetromino_id-1]
        self.tetromino_color = self.tetromino_id
        self.tetromino_offset = [-2, FIELD_WIDTH//2]
        self.gameOver = not any(self.is_cell_free(r, c) for (r, c) in self.get_tetromino_coords())
    
    def get_tetromino_coords(self):
        return [(r+self.tetromino_offset[0], c + self.tetromino_offset[1]) for (r, c) in self.tetromino]

    def apply_tetromino(self):
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
        return self.tetromino_color if (r, c) in self.get_tetromino_coords() else self.field[r][c]
    
    def is_cell_free(self, r, c):
        return r < FIELD_HEIGHT and 0 <= c < FIELD_WIDTH and (r < 0 or self.field[r][c] == 0)
    
    def move(self, dr, dc):
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
            return

class GameApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.epsilon = 0.0 # randomness ratio (0.0 ~ 1.0) 0.0 = deterministic, 1.0 = totally random
        self.gamma = 0.9   # discount rate for value calculation in Reinforcement Learning
        self.lr = 0.001 # learning rate
        self.oreward = 0.0
        self.nreward = 0.0
        self.autoLearning = False
        self.tetris = Tetris()
        self.agent = Agent()
        self.agent.onNewGame()
        self.model = GameNet() # output_state: none, left, right, up, down
        self.trainer = GameTrainer(self.model, self.lr, self.gamma, self.oreward, self.nreward)
        self.parallelWorkers = []
        for _ in range(max(0, PARALLEL_TASK_COUNT - 1)):
            worker_agent = Agent()
            worker_agent.onNewGame()
            self.parallelWorkers.append({"tetris": Tetris(), "agent": worker_agent})
        self.latestLoss = 0.0
        self.consoleRenderTick = 0
        self.consoleRenderEvery = 8
        self.create_widgets()
        self.pack()
        self.update_clock()
        self.mapState = None


 
    def update_clock(self):
        self.tetris.move(1, 0)
        self.update()  
        #self.master.after(int(1000*(0.66**self.tetris.level)), self.update_clock)
        if(self.autoLearning):
            self.master.after(1, self.update_clock)
        else:
            self.master.after(660, self.update_clock)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, height= PIECE_SIZE*FIELD_HEIGHT, 
                                      width = PIECE_SIZE*FIELD_WIDTH, bg="black", bd=0)
        self.canvas.bind('<Left>',   lambda _: (self.tetris.move(0, -1), self.update()))
        self.canvas.bind('<Right>',  lambda _: (self.tetris.move(0, 1), self.update()))
        self.canvas.bind('<Down>',   lambda _: (self.tetris.move(1, 0), self.update()))
        self.canvas.bind('<Up>',     lambda _: (self.tetris.rotate(), self.update()))
        #self.canvas.bind('<Return>', lambda _: self.tetris.restart())
        #self.canvas.bind('<Escape>', lambda _: sys.exit(0))
        self.canvas.focus_set()
        self.rectangles = [
            self.canvas.create_rectangle(c*PIECE_SIZE, r*PIECE_SIZE, (c+1)*PIECE_SIZE, (r+1)*PIECE_SIZE)
                for r in range(FIELD_HEIGHT) for c in range(FIELD_WIDTH)
            ]
        self.canvas.pack(side="left")

        self.status_msg = tk.Label(self, anchor='w', width=20, font=("Courier", 24, 'bold'))
        self.status_msg.pack(side="top")

        self.block = tk.Canvas(self, height = BLOCK_SIZE*4, width = BLOCK_SIZE*2, bd=0)
        self.brect = [
            self.block.create_rectangle(c*BLOCK_SIZE, r*BLOCK_SIZE, (c+1)*BLOCK_SIZE, (r+1)*BLOCK_SIZE)
                for r in range(4) for c in range(2)
           ]
        self.block.pack(side="top")

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
        if(self.autoLearning==False):
            self.model.loadGameNet()
            self.autoLearning = True
            self.autoLearnButton.configure(state=tk.ACTIVE, relief='groove')
            self.manualLearnButton.configure(state=tk.NORMAL, relief='raised')
    
    def manualLearningGame(self):
        if(self.autoLearning==True):
            self.autoLearning = False
            self.manualLearnButton.configure(state=tk.ACTIVE, relief='groove')
            self.autoLearnButton.configure(state=tk.NORMAL, relief='raised')

    def saveModel(self):
        saved_path = self.model.saveGameNet()
        messagebox.showinfo('Save Network', f'Saved model:\n{saved_path}')

    def loadModel(self):
        model_folder = os.path.abspath('./weight')
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
        if(move == 1):
            tetris_obj.move(0, -1)
        elif(move == 2):
            tetris_obj.move(0, 1)
        elif(move == 3):
            tetris_obj.rotate()
        elif(move == 4):
            tetris_obj.move(1, 0)

    def _run_train_step(self, tetris_obj, agent_obj, map_state=None):
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
        # clear screen
        for i, _id in enumerate(self.brect):
                self.block.itemconfig(_id, fill='white')
        for (r, c) in TETROMINOS[self.tetris.next_tetromino_id-1]:
            _id = 2 * r + c + 1
            self.block.itemconfig(_id, fill=COLORS[self.tetris.next_tetromino_id])

        mylist = []
        for i, _id in enumerate(self.rectangles):
            self.color_id = self.tetris.get_color(i//FIELD_WIDTH, i % FIELD_WIDTH)
            self.canvas.itemconfig(_id, fill=COLORS[self.color_id])
            mylist.append(self.color_id)

        mystring = [str(x) for x in mylist]
        statelist  = mylist
        outstring = ""
        for i in range(len(mystring)):
            outstring += mystring[i]
            if(((i+1)%FIELD_WIDTH)==0):
                outstring += "\n"
        outstring += "\n"


        total_games = self.agent.totalGames + sum(w["agent"].totalGames for w in self.parallelWorkers)
        self.status_msg['text'] = "Score:%05d\nLevel:%02d\nGames:%05d\nTasks:%02d\nEps:%0.2f\nLoss:%0.2f"%(
            self.tetris.score,
            self.tetris.level,
            total_games,
            PARALLEL_TASK_COUNT,
            self.agent.epsilon if hasattr(self.agent, 'epsilon') else 1.0,
            self.latestLoss,
        )
        self.mapState_msg['text'] = outstring

        if CONSOLE_RENDER:
            self.consoleRenderTick += 1
            if self.consoleRenderTick % self.consoleRenderEvery == 0:
                self._render_console_board()

        if self.tetris.gameOver:
            self.restartButton.configure(state=tk.ACTIVE)
        else:
            self.restartButton.configure(state=tk.DISABLED)
        
        self.mapState = self.agent.createMapState(statelist)
        self.rewardFactors = self.agent.createRewardFactors(self.mapState)

        #if __debug__:
        #    first_rlist, bump_rlist, height_rlist, piece_rlist, hole_rlist, line_clist = self.rewardFactors
        #    print(first_rlist)
        #    print(bump_rlist)
        #    print(height_rlist)
        #    print(piece_rlist)
        #    print(hole_rlist)
        #    print(line_clist)
        #    print(self.mapState)

        self.nreward = 0.0
        if(self.autoLearning):
            losses = []
            main_loss, train_reward = self._run_train_step(self.tetris, self.agent, self.mapState)
            losses.append(main_loss)
            self.nreward = train_reward

            for worker in self.parallelWorkers:
                worker["tetris"].move(1, 0)
                worker_loss, _ = self._run_train_step(worker["tetris"], worker["agent"])
                losses.append(worker_loss)

            self.latestLoss = float(np.mean(losses)) if losses else 0.0
     
        # when update() is called again by the Canvas mainloop 
        # the self.mapState and self.rewardFactors will be updated

        #pvState, pvMove, pvReward, pvLoss = self.agent.pvAction(self.mapState)

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

    root.update_idletasks()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    pos_x = (screen_width - window_width) // 2
    pos_y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{pos_x}+{pos_y}')
    
    game.mainloop() 