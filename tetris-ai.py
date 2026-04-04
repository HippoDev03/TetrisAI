import os
import sys
import random
import tkinter as tk
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
SCORE_PER_ELIMINATED_LINES = (0, 40, 100, 300, 1200)

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

        x = torch.tensor(state, dtype=torch.float)
        x = torch.flatten(x)
        x = self.linear1(x)
        # x = F.relu(x)
        x = self.linear2(x)
        # x = F.relu(x)
        x = self.linear3(x)
        # x = F.relu(x)
        x = self.linear4(x)
        return x


    def saveGameNet(self, file_name='GameQNet.wgt'):
        model_folder_path = './weight'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def loadGameNet(self, file_name='GameQNet.wgt'):
         model_folder_path = './weight'
         file_name = os.path.join(model_folder_path, file_name)
         try:
             self.load_state_dict(torch.load(file_name))
         except:
             print("Error: Unable to load state dictionary: {}".format(file_name))
         self.eval()


class GameTrainer:
    def __init__(self, model, lr, gamma, oreward, nreward):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.oreward = oreward
        self.nreward = nreward
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss(oreward, nreward)

    def trainStep(self, oreward, nreward):
        # run QNet
        self.optimizer.zero_grad()
        xx = torch.tensor([1000000.0])
        yy = torch.tensor(nreward)
        lossfn = self.criterion(xx, yy)
        #lossfn.backward()

        self.optimizer.step()


class Agent():
    def __init__(self):
        self.totalGames = 0
        self.maxMemory = 100_000  # max memory
        self.batchSize = 1_000 # max batch training size
        self.memory = deque(maxlen=self.maxMemory)


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
        self.epsilon = 10 - self.totalGames
        finalMove = [0, 0, 0, 0, 0] # None, left, right, up, down
        if random.randint(0, 5) < self.epsilon:
            move = random.randint(0,4) # 0, 1, 2, 3, 4
            finalMove[move] = 1
        else:
            predAction = model(state)
            move = torch.argmax(predAction).item()
            # print(predAction)
            finalMove[move]=1

        return finalMove
    
    def getReward(self, myRewardFactors, myEliminated):
        first_rlist, bump_rlist, height_rlist, piece_rlist, hole_rlist, line_clist = myRewardFactors
        
        list_frl = (100 * first_rlist);    
        list_brl = (-26 * bump_rlist);            
        list_hrl = (-20 * height_rlist);          
        list_prl = (-25 * piece_rlist);               
        list_orl = (-20 * hole_rlist);            
        
        lcl = 0.0
        for i in range(len(line_clist)):
            lcl += 100.0 / (float(FIELD_WIDTH - line_clist[i]) + 1.0)

        frl = list_frl.sum()
        brl = list_brl.sum()
        hrl = list_hrl.sum()
        prl = list_prl.sum() 
        orl = list_orl.sum()
        ell = myEliminated * 500.0

        reward  = lcl + frl +brl + hrl + prl + orl + ell
        print(reward)
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
            return

        if all(self.is_cell_free(r + dr, c + dc) for (r, c) in self.get_tetromino_coords()):
            self.tetromino_offset = [self.tetromino_offset[0] + dr, self.tetromino_offset[1] + dc]
        elif dr == 1 and dc == 0:
            self.gameOver = any(r < 0 for (r, c) in self.get_tetromino_coords())
            if not self.gameOver:
                self.apply_tetromino()

    def rotate(self):
        if self.gameOver:
            return

        ys = [r for (r, c) in self.tetromino]
        xs = [c for (r, c) in self.tetromino]
        size = max(max(ys) - min(ys), max(xs) - min(xs))
        rotated_tetromino = [(c, size-r) for (r, c) in self.tetromino]
        wallkick_offset = self.tetromino_offset[:]
        tetromino_coord = [(r+wallkick_offset[0], c + wallkick_offset[1]) for (r, c) in rotated_tetromino]
        min_x = min(c for r, c in tetromino_coord)
        max_x = max(c for r, c in tetromino_coord)
        max_y = max(r for r, c in tetromino_coord)
        wallkick_offset[1] -= min(0, min_x)
        wallkick_offset[1] += min(0, FIELD_WIDTH - (1 + max_x))
        wallkick_offset[0] += min(0, FIELD_HEIGHT - (1 + max_y))
        tetromino_coord = [(r+wallkick_offset[0], c + wallkick_offset[1]) for (r, c) in rotated_tetromino]
        if all(self.is_cell_free(r, c) for (r, c) in tetromino_coord):
            self.tetromino, self.tetromino_offset = rotated_tetromino, wallkick_offset
    
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
        self.model = GameNet() # output_state: none, left, right, up, down
        self.trainer = GameTrainer(self.model, self.lr, self.gamma, self.oreward, self.nreward)
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

        self.mapState_msg = tk.Label(self, anchor='center', width=11, font=("Courier", 12, 'bold'), fg='darkblue')
        self.mapState_msg.pack(side="top")

        self.saveButton = tk.Button(self, text='Save Network', anchor='center', width=12, 
                                    activeforeground="red", command=self.model.saveGameNet)
        self.saveButton.pack(side='top')

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


        self.status_msg['text'] = "Score:%05d\nLevel:%02d\nLearning:%05d"%(self.tetris.score, self.tetris.level, self.agent.totalGames)
        self.mapState_msg['text'] = outstring

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

        # reward for current move 
        self.nreward = self.agent.getReward(self.rewardFactors, self.tetris.lines_eliminated)

        # training
        #error = self.nreward
        #print("Error: ", error)

        self.trainer.trainStep(self.oreward, self.nreward)
        self.oreward = self.nreward

        
        # new move based on old rewardFactors 
        if(self.autoLearning):
            finalMove = self.agent.getAction(self.model, self.mapState)

            # brick movement main body
            if(finalMove == [0, 1, 0, 0, 0]):
                    self.tetris.move(0, -1) 
            elif(finalMove == [0, 0, 1, 0, 0]):
                    self.tetris.move(0, 1) 
            elif(finalMove == [0, 0, 0, 1, 0]):
                    self.tetris.rotate() 
            elif(finalMove == [0, 0, 0, 0, 1]):
                    self.tetris.move(1, 0) 
     
        # when update() is called again by the Canvas mainloop 
        # the self.mapState and self.rewardFactors will be updated

        #pvState, pvMove, pvReward, pvLoss = self.agent.pvAction(self.mapState)

        if self.tetris.gameOver:
            self.agent.totalGames += 1
            self.tetris.restart()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Deep Learning Tetris Game by Daniel Chen v2')
    root.iconbitmap('icon.ico')
    game = GameApp(master=root)
    
    game.mainloop() 