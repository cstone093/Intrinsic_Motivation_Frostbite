import numpy as np

class Base_Q_Value_Function:
    def __init__(self,hyp,a_size,s_size):
        self.hyp = hyp
        self.A_SIZE = a_size
        self.S_SIZE = s_size
        self.learning_rate = hyp["INIT_LEARNING_RATE"]
        np.random.seed(self.hyp["SEED"])

        self.eps_gradient_1 = (
            -(self.hyp["EPS_STEPS_INIT"] - self.hyp["EPS_STEPS_INTER"]) / self.hyp["EPS_FRAMES_INTER"]
        )
        self.eps_intercept_1 = self.hyp["EPS_STEPS_INIT"] - self.eps_gradient_1 * self.hyp["EPS_FRAMES_INIT"]
        
        self.eps_gradient_2 = -(self.hyp["EPS_STEPS_INTER"] - self.hyp["EPS_STEPS_FINAL"]) / (
            self.hyp["EPS_FRAMES_FINAL"]- self.hyp["EPS_FRAMES_INTER"] - self.hyp["EPS_FRAMES_INIT"]
        )
        self.eps_intercept_2 = self.hyp["EPS_STEPS_FINAL"] - self.eps_gradient_2 * self.hyp["EPS_FRAMES_FINAL"]
    # Creates a CNN for the policy
    def _create_CNN(self):
        raise(NotImplementedError)

    def get_epsilon(self,frames):
        if frames < self.hyp["EPS_FRAMES_INIT"]:
            return self.hyp["EPS_STEPS_INIT"]
        elif frames < self.hyp["EPS_FRAMES_INIT"] + self.hyp["EPS_FRAMES_INTER"]:
            return self.eps_gradient_1 * frames + self.eps_intercept_1
        else:
            return self.eps_gradient_2 * frames + self.eps_intercept_2

    # Takes a batch of experience and performs back propagation on the CNN
    def learn(self,states,actions,rewards,new_states,terminals):
        raise(NotImplementedError)

    # Given a state, uses the CNN as a policy to choose an action
    def choose_action(self,state,frame):
        raise(NotImplementedError)

    def update_target_model(self):
        raise(NotImplementedError)

    def save_state(self,directory):
        raise(NotImplementedError)

    def load_state(self,directory):
        raise(NotImplementedError)