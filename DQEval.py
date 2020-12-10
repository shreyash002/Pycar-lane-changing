from dqn import weights_init, Initializer, DQN, HuberLoss, ReplayMemory
from pycar_env import PyCar
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from collections import namedtuple

import sys
import math
import random
import shutil
import warnings
from torch.backends import cudnn
from tqdm import tqdm

import logging
import time
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DQNAgent:

    def __init__(self):
        # self.config = config
        self.gamma = 0.4

        # self.logger = logging.getLogger("DQNAgent")

        self.screen_width = 600

        # define models (policy and target)
        self.policy_model = DQN()
        self.target_model = DQN()

        # define memory
        self.memory = ReplayMemory()

        # define loss
        self.loss = HuberLoss()

        # define optimizer
        self.optim = torch.optim.Adam(self.policy_model.parameters(), lr=0.01)

        # define environment
        self.env = PyCar()#TODO
        # self.cartpole = PyCar(self.screen_width)

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        self.batch_size = 1700

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()


        self.cuda = self.is_cuda 

        if self.cuda:
            # print_cuda_statistics()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Initialize Target model with policy model state dict
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.savepath = "/home/sk002/Desktop/model/"


    def run(self, num):
        """
        This function will the operator
        :return:
        """
        self.policy_model.load_state_dict(torch.load(self.savepath+"policy_epoch"+num+".pth"))
        self.target_model.load_state_dict(torch.load(self.savepath+"target_epoch"+num+".pth"))
        try:
            self.validate()

        except KeyboardInterrupt as e:
            print(e)
            #self.logger.info("You have entered CTRL+C.. Wait to finalize")



    def get_action(self, state):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """

        if self.cuda:
            state = state.cuda()
        with torch.no_grad():
            return self.policy_model(state).max(1)[1].view(1, 1)  # size (1,1)
     
    def validate(self):
        

        total = 200
        reward_total = 0
        for episode in range(total):

            self.env.reset_game()

            curr_state = torch.Tensor(self.env.get_state()).permute(2, 0, 1).unsqueeze(0)

            while(1):
                # time.sleep(0.1)

                #episode_duration += 1
                # select action
                action = self.get_action(curr_state)
                # perform action and get reward
                # print(action)
                images, reward, done,score = self.env.step(action.item())#TODO

                if self.cuda:
                    reward = torch.Tensor([reward]).to(self.device)
                else:
                    reward = torch.Tensor([reward]).to(self.device)

    
                # assign next state
                if done:
                    next_state = None
                else:
                    next_state = torch.Tensor(images).permute(2, 0, 1).unsqueeze(0) #TODO

                curr_state = next_state
                
                if done:
                    reward_total+=score
                    # print(score)
                    break

        print(reward_total/total)
        # pass

if __name__=="__main__":

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    warnings.filterwarnings("ignore", category=UserWarning)

    agent = DQNAgent()
    agent.run(sys.argv[1])
    
