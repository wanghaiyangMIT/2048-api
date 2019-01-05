import numpy as np
from models import MyAgent
import torch
import json
import torchvision.transforms as transforms
import os

normalize = transforms.Normalize(mean=[.5], std=[.5])  
transform = transforms.Compose([transforms.ToTensor(), normalize])
    
class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        self.iter = 0

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            self.iter = n_iter
            if verbose:
                #print("Iter: {}".format(n_iter))
                #print("======Direction: {}======".format(
                #    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyTestAgent(Agent):

    def __init__(self,game,time,save_dir,data_dir,model,args = None,display = None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game,display)
        from expectimax import board_to_move
        self.model = MyAgent(args,batch_size = 1).cuda().eval()
        self.search_func = board_to_move
        self.time = time
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.model = model

    #def load_model(self):
        #self.model = self.model.cuda()
        #self.model.load_state_dict(torch.load(os.path.join(self.save_dir,'last_params.pkl')))
        #a = self.model.Resnet34.layer3.state_dict()
        #for key,v in a.items():
            #print(key,v)
        
        #for key, v in self.model.items():
            #print (key, v)
    def step(self):
        oriagentpre = self.search_func(self.game.board)
        game_board = self.game.board
        baddata = {'tabel':game_board.tolist(),'direction':oriagentpre}
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        with open (os.path.join(self.data_dir,str(self.time)+'_'+str(self.iter)+'_'+str(self.game.score)+'.json'),'w') as f:
            bad_data = json.dumps(baddata)
            f.write(bad_data)
        game_board = np.expand_dims(game_board,axis = 2)
        game_board = transform(game_board).cuda().float().view(1,1,4,4)
        pred = self.model(game_board)
        pre = torch.max(pred,1)[1]
        direction = int(pre)
        #print(pre)
        #print(oriagentpre,direction)

        return direction
