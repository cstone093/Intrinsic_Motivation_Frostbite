##########################################
# Forward model super class
##########################################
import numpy as np
from datetime import datetime 
from PIL import Image as im

import os



class ForwardModel():
    def __init__(self,hyp,a_size,s_size):
        self.hyp = hyp
        self.A_SIZE = a_size
        self.STATE_SHAPE = s_size
        self.learning_rate = hyp["INIT_LEARNING_RATE"]
        np.random.seed(self.hyp["SEED"])

        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")
        self.pred_dir = os.path.join(self.log_dir, "predictions")

    def _save_prediction(self,pred):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        data = pred[0][:,:,0].astype(np.uint8)
        image = im.fromarray(data)
        image.save(f'{self.pred_dir}/prediction{current_time}.png')

    def _create_CNN(self):
        raise(NotImplementedError)

    def learn(self,obs,acs,new_obs):
        raise(NotImplementedError)

    def get_error(self,s,a,new_s,training=True):
        raise(NotImplementedError)