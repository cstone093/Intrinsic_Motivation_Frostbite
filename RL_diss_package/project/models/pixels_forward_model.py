from project.models.forward_model import ForwardModel
import tensorflow as tf
import numpy as np


from keras import Model
from keras.layers import  Dense, Activation, Conv2D, Flatten, Rescaling, Reshape, concatenate
from keras.layers import Input
from keras.optimizers import adam_v2


class FM_Pixels(ForwardModel):

    def __init__(self,hyp,a_size,s_size):
        super(FM_Pixels,self).__init__(hyp=hyp,a_size=a_size,s_size=s_size)    
        print("Setting up pixels-based intrinsic method")
        self.model = self._create_CNN()

        print("Forward Model Initialised:")
        self.model.summary()

        self.lowest_error = np.infty

    def _create_CNN(self):
        initializer = tf.keras.initializers.HeNormal

        # Convolutional branch with the state
        state_input = Input(shape=self.STATE_SHAPE)
        state_branch = Rescaling(scale=1.0/255)(state_input)

        state_branch = Conv2D(32,(8, 8),strides=4,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)

        state_branch = Conv2D(64,(4, 4),strides=2,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)

        state_branch = Conv2D(64,(3, 3),strides=1,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)

        state_branch = Flatten()(state_branch)
        state_branch = Model(inputs=state_input,outputs=state_branch)

        # Chosen to be class as no linearity of action representation
        action_input = Input(shape=self.A_SIZE)
        action_branch = Dense(self.A_SIZE)(action_input)
        action_branch = Model(inputs=action_input,outputs=action_branch)

        # Combine with action
        s_a_pair = concatenate([state_branch.output,action_branch.output])

        encoder = Dense(512,kernel_initializer=initializer,activation="relu")(s_a_pair)
        encoder = Dense(512,kernel_initializer=initializer,activation="relu")(encoder)
        encoder = Dense(np.prod(self.STATE_SHAPE),kernel_initializer=initializer,activation="relu")(encoder)
        encoder = Reshape(self.STATE_SHAPE)(encoder)

        model = Model(inputs=[state_input,action_input],outputs=encoder)

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model
    
    # Learn from batch of transitions
    def learn(self,obs,acs,new_obs):
        actions = np.zeros(shape=(acs.shape[0],self.A_SIZE),dtype=np.uint8)
        for i,a in enumerate(acs):
            actions[i][a] = 1
        training_in=[obs,actions]
        self.model.fit(
            training_in,
            new_obs,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )

    # Given state, action and next state, calculate and return prediction error
    def get_error(self,s,a,new_s,training=True):
        action = np.zeros(shape=(1,self.A_SIZE),dtype=np.uint8)
        action[0][a] = 1
        s = np.array([s],dtype=np.float32)
        prediction = self.model.predict([s,action],verbose=0)

        L2_error = np.linalg.norm(prediction-new_s)
        if training:
            self._save_if_lowest_error(L2_error,prediction)
        return L2_error

    # Save image of prediction if most accurate so far
    def _save_if_lowest_error(self,L2_error,prediction):
        if L2_error < self.lowest_error:
            self.lowest_error = L2_error
            self._save_prediction(prediction)
        return
