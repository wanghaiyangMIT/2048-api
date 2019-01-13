import json
import os
from tqdm import tqdm
import time
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import MyAgent
from utils import collate_fn
from dataloader import mydataset

from tqdm import tqdm
from game import Game
from displays import Display

parser = argparse.ArgumentParser(description='2048')
parser.add_argument('--data_dir',   type = str  , default = '')
parser.add_argument('--test_dir',   type = str  , default = '')
parser.add_argument('--save_dir',   type = str  , default = '')
parser.add_argument('--batch_size', type = int  , default = 1 )
parser.add_argument('--device',     type = int  , default=-1, help='gpu device id')
parser.add_argument('--lr',         type = float, default=0.00003)
parser.add_argument('--num_epoch',  type=int,   default=200,  help='maximum number of epochs')
parser.add_argument('--scratchtrain',  type=int,   default=1,  help='train from scratch,1 means train from scrach, 0 means train from pre_params.')
args = parser.parse_args()

if len(args.save_dir) == 0:
    args.save_dir = 'train'
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
else:
    print('save_dir ',args.save_dir,'  is exists,print yes to confirm')
    s = input()
    if 'yes'not in s:
        assert False

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)
    print('data_dir has been created')
def handle_batch(args, model,batch):
    tabel, direction = batch
    with torch.no_grad():
        tabel = tabel.cuda().float()
        tabel = tabel.view(args.batch_size,1,4,4)
        direction = direction.cuda().long().view(args.batch_size)
    pred = model(tabel)         
    return tabel,direction, pred

def single_run(size, score_to_win, AgentClass,time,save_dir, data_dir, model,**kwargs):
    game = Game(size, score_to_win)
    #agent = AgentClass(game, display=Display(), **kwargs)
    agent = AgentClass(game,time,save_dir = save_dir,data_dir = data_dir,model = model, display = None, **kwargs)
    agent.play(verbose=True)
    return game.score

if __name__ == '__main__':
    if args.device > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
        model = MyAgent(args,args.batch_size).cuda()
        if agrs.scratchtrain == 0:
            model.load_state_dict(torch.load('pretrain_params.pkl'))
    adam = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    run_best_score = 0
    for epoch in range(args.num_epoch):
        #change  temp data 500test
        if epoch % 40 == 0:
            #clean data
            datalist = os.listdir(args.data_dir)
            if len(datalist) != 0:
                for data in datalist:
                    datafile = os.path.join(args.data_dir,data)
                    os.remove(datafile)
                if len(os.listdir(args.data_dir)) == 0:
                    print('clean all the data')
                    with open(os.path.join(args.save_dir,'train_log.txt'),'a') as f:
                        f.write('\n clean all the data')

            GAME_SIZE = 4
            SCORE_TO_WIN = 65563
            N_TESTS = 1000
            START_N_TESTS = 0
            END_N_TESTS = N_TESTS
            
            model.eval()
            from myagents import MyTestAgent as TestAgent
            scores = []
            for times in tqdm(range(START_N_TESTS,END_N_TESTS)):
                score = single_run(GAME_SIZE, SCORE_TO_WIN,
                                     AgentClass=TestAgent,time = times,save_dir = args.save_dir,data_dir = args.data_dir,model = model)
                scores.append(score)

            test_run_report = "\n Generateing data : Average scores:"+str(N_TESTS)+"times" +'_'+ str(sum(scores) / len(scores))
            print(test_run_report)
            with open(os.path.join(args.save_dir,'train_log.txt'),'a') as f:
                f.write(test_run_report)

        traindataset = mydataset(args.data_dir)
        traindataloader = DataLoader(traindataset,args.batch_size,shuffle = True,collate_fn = collate_fn,drop_last = True)
        '''
        testdataset = mydataset(args.test_dir)
        testdataloader = DataLoader(testdataset,args.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
        '''

        running_loss = 0
        running_acc = 0
        batchiter = 0
        model.train()
        for batch in tqdm(traindataloader):
            tabel,direction,pred = handle_batch(args,model,batch)
            adam.zero_grad()
            loss = model.loss(tabel,direction)
            batchiter += 1
            loss.backward()
            adam.step()
            running_loss += loss.item()
            valu,pre = torch.max(pred,1)
            pre = pre.float()
            direction = direction.view(args.batch_size).cuda().float()
            correct_num = (pre==direction).sum()
            running_acc += correct_num.item()
        running_loss /= len(traindataset)
        running_loss = running_loss*10000
        running_acc /= len(traindataset)
        train_report = "\n train [%d/%d] Loss: %7f, Acc: %.3f"%(epoch+1,args.num_epoch,running_loss,
                                           100*running_acc)
        print(train_report)
        with open(os.path.join(args.save_dir,'train_log.txt'),'a') as f:
            f.write(train_report)
        if epoch%500 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, str(epoch+1)+'_params.pkl'))
        torch.save(model.state_dict(), os.path.join(args.save_dir,'last_params.pkl'))
        
        '''
        #test in dir
        running_acc = 0
        model.eval()
        for batch in tqdm(testdataloader):
            tabel,direction,pred = handle_batch(args,model,batch)
            valu,pre = torch.max(pred,1)
            pre = pre.float()
            direction = direction.view(args.batch_size).cuda().float()
            print(direction[0],pre[0])
            correct_num = (pre==direction).sum()
            running_acc += correct_num.item()
        running_acc /= len(testdataset)
        test_report = "\n Test [%d/%d] , Acc: %.3f"%(epoch+1,args.num_epoch,
                                           100*running_acc)
        print(test_report)
        with open(os.path.join(args.save_dir,'train_log.txt'),'a') as f:
            f.write(test_report)
        '''
        if (epoch+1)%40 == 0:
            GAME_SIZE = 4
            SCORE_TO_WIN = 65563
            N_TESTS = 50
            START_N_TESTS = 10000
            END_N_TESTS = 10000 + N_TESTS
            
            model.eval()
            from testagents import MyTestAgent as TestAgent
            scores = []
            for times in tqdm(range(START_N_TESTS,END_N_TESTS)):
                score = single_run(GAME_SIZE, SCORE_TO_WIN,
                                     AgentClass=TestAgent,time = times,save_dir = args.save_dir, data_dir = args.data_dir, model = model)
                scores.append(score)

            test_run_report = "\n Average scores:"+str(N_TESTS)+"times" +'_'+ str(sum(scores) / len(scores))
            print(test_run_report)
            print('run_best_score',run_best_score)
            with open(os.path.join(args.save_dir,'train_log.txt'),'a') as f:
                f.write(test_run_report)
                f.write('\n run_best_score = '+str(run_best_score))
            temp_score = sum(scores) / len(scores)
            if (temp_score > run_best_score):
                run_best_score = temp_score
                torch.save(model.state_dict(), os.path.join(args.save_dir, str(run_best_score)+'_params.pkl'))
  











