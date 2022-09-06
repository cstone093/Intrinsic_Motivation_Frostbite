import numpy as np

import tensorflow as tf
from keras import Model
from keras.layers import Dense, Activation, Conv2D, Flatten, Rescaling, concatenate
from keras.layers import Input
from keras.optimizers import adam_v2

from project.models.forward_model import ForwardModel

class FM_Random(ForwardModel):

    def __init__(self,hyp,a_size,s_size,feature_count):
        super(FM_Random,self).__init__(hyp=hyp,a_size=a_size,s_size=s_size)
        print("Setting up random intrinsic network")
        self.feature_count = feature_count
        self.feature_net = self._create_features_CNN()
        print("Created random feature extractor network:")
        self.feature_net.summary()
        self.model = self._create_CNN()
        print("Created model network:")
        self.model.summary()

    def _create_features_CNN(self):
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

        # Output of features
        # Need to look into feature count
        features = Dense(512,kernel_initializer=initializer)(state_branch) # For linearity 

        model = Model(inputs=[state_input],outputs=features)

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model

    def _create_CNN(self): 
        initializer = tf.keras.initializers.HeNormal

        # Convolutional branch with the state
        feature_input = Input(shape=self.feature_count)
        feature_branch = Dense(self.feature_count,kernel_initializer=initializer,activation="relu")(feature_input)
        feature_branch = Model(inputs=feature_input,outputs=feature_branch)

        # Chosen to be class as no linearity of action representation
        action_input = Input(shape=self.A_SIZE)
        action_branch = Dense(self.A_SIZE,kernel_initializer=initializer,activation="relu")(action_input)
        action_branch = Model(inputs=action_input,outputs=action_branch)

        # Combine with action
        s_a_pair = concatenate([feature_branch.output,action_branch.output])

        dense_layer = Dense(self.feature_count,kernel_initializer=initializer,activation="relu")(s_a_pair)
        dense_layer = Dense(self.feature_count,kernel_initializer=initializer,activation="relu")(dense_layer)
        predicted_features = Dense(self.feature_count,kernel_initializer=initializer)(dense_layer)

        model = Model(inputs=[feature_input,action_input],outputs=predicted_features)

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model
    
    def learn(self,obs,acs,new_obs):
        obs_features = self.feature_net.predict(obs,verbose=0)
        new_obs_features = self.feature_net.predict(new_obs,verbose=0)

        actions = np.zeros(shape=(acs.shape[0],self.A_SIZE),dtype=np.uint8)
        for i,a in enumerate(acs):
            actions[i][a] = 1

        training_in=[obs_features,actions]
        self.model.fit(
            training_in,
            new_obs_features,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )

    def get_error(self,s,a,new_s,training=False):
        action = np.zeros(shape=(1,self.A_SIZE),dtype=np.uint8)
        action[0][a] = 1

        s = np.array([s],dtype=np.float32)
        s_feat = self.feature_net.predict([s],verbose=0)

        pred_new_s_feat = self.model.predict([s_feat,action],verbose=0)

        new_s = np.array([new_s],dtype=np.float32)
        new_s_feat = self.feature_net.predict([new_s],verbose=0)[0]
        diff = pred_new_s_feat[0] - new_s_feat[0]
        mse = ((diff)**2).mean(axis=0)
        return mse
