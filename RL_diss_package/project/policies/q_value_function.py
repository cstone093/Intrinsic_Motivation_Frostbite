
import os
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv2D, Flatten, Rescaling
from keras.optimizers import adam_v2

from project.policies.base_q_value_function import Base_Q_Value_Function

class Q_Value_Function(Base_Q_Value_Function):
    def __init__(self,hyp,a_size,s_size):
        super(Q_Value_Function,self).__init__(hyp=hyp,a_size=a_size,s_size=s_size)

        self.local_model = self._create_CNN()
        self.target_model = self._create_CNN()

        self.update_target_model()

        self.local_model.summary()
        self.target_model.summary()
        
    # Creates a CNN for the policy
    def _create_CNN(self):
        initializer = tf.keras.initializers.HeNormal

        model = Sequential()

        model.add(InputLayer(input_shape=self.S_SIZE))

        model.add(Rescaling(scale=1.0/255))

        model.add(Conv2D(32,(8, 8),strides=4,kernel_initializer=initializer))
        model.add(Activation("relu"))

        model.add(Conv2D(64,(4, 4),strides=2,kernel_initializer=initializer))
        model.add(Activation("relu"))

        model.add(Conv2D(64,(3, 3),strides=1,kernel_initializer=initializer))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=initializer,activation="relu")) # Changed 

        model.add(Dense(self.A_SIZE, activation="linear",kernel_initializer=initializer))

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model

    # Takes a batch of experience and performs back propagation on the CNN
    def learn(self,states,actions,rewards,new_states,terminals):

        # Get predicted Q values for each s using our current model
        curr_q_values = self.local_model.predict(states,verbose=0)
        # Get Q values for each new_s using the target model
        future_q = self.target_model.predict(new_states,verbose=0)

        updated_q_values = rewards + self.hyp["GAMMA"] * np.max(future_q, axis=1) * (
            1 - terminals
        )

        for index, (action, new_q) in enumerate(zip(actions, updated_q_values)):
            curr_q_values[index][action] = new_q

        # Fit the model using our corrected values.
        self.local_model.fit(
            states,
            curr_q_values,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )

    # Given a state, uses the CNN as a policy to choose an action
    def choose_action(self,state,frame):
        # do epsilon soft
        if np.random.uniform(0,1) <= self.get_epsilon(frame):
            return np.random.randint(self.A_SIZE)
        else:
            values = self.local_model.predict(state.reshape(-1, *state.shape),verbose=0)
            return np.argmax(values[0])
            
    def update_target_model(self):
        self.target_model.set_weights(self.local_model.get_weights())

    def save_state(self,directory):
        filename = os.path.join(directory, "weights.hdf5")
        self.local_model.save_weights(filename)
        print(f"Agent network weights were saved in: {filename}")

    def load_state(self,directory):
        filename = os.path.join(directory, "weights.hdf5")
        self.local_model.load_weights(filename)
        self.update_target_model()
        print(f"Agent network weights were loaded from: {filename}")
