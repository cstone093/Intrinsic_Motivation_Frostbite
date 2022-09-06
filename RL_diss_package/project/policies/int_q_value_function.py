
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv2D, Flatten, Rescaling
from keras.optimizers import adam_v2
import os

from project.policies.q_value_function import Base_Q_Value_Function

class Int_Q_Value_Function(Base_Q_Value_Function):
    def __init__(self,hyp,a_size,s_size):
        super(Int_Q_Value_Function,self).__init__(hyp=hyp,a_size=a_size,s_size=s_size)
        
        print("Setting up extrinsic Q value function")
        self.int_local_model = self._create_CNN()
        self.int_target_model = self._create_CNN()
        self.int_local_model.summary()

        print("Setting up extrinsic Q value function")
        self.ext_local_model = self._create_CNN()
        self.ext_target_model = self._create_CNN()
        self.ext_local_model.summary()
        
        self.update_target_model()
        
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
    def learn(self,states,actions,int_rewards,ext_rewards,new_states,terminals):
        # Get local model Q values for states
        ext_curr_q_values = self.ext_local_model.predict(states,verbose=0)
        # Get target model Q values for new states
        ext_future_q = self.ext_target_model.predict(new_states,verbose=0)
        # Update the q values using Bellman Eq.
        ext_updated_q_values = ext_rewards + self.hyp["GAMMA"] * np.max(ext_future_q, axis=1) * (
            1 - terminals
        )

        int_curr_q_values = self.int_local_model.predict(states,verbose=0)
        int_future_q = self.int_target_model.predict(new_states,verbose=0)

        int_updated_q_values = int_rewards + self.hyp["GAMMA"] * np.max(int_future_q, axis=1) * (
            1 - terminals
        )

        for index, (action, ext_new_q, int_new_q) in enumerate(zip(actions, ext_updated_q_values, int_updated_q_values)):
            ext_curr_q_values[index][action] = ext_new_q
            int_curr_q_values[index][action] = int_new_q

        # Fit the extrinsic model using our corrected values.
        self.ext_local_model.fit(
            states,
            ext_curr_q_values,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )
        
        # Fit the intrinsic model using our corrected values.
        self.int_local_model.fit(
            states,
            int_curr_q_values,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )

    def combine_q(self,state,alpha,beta):
        int_values = self.int_local_model.predict(state.reshape(-1, *state.shape),verbose=0)
        ext_values = self.ext_local_model.predict(state.reshape(-1, *state.shape),verbose=0)

        # Using identity as suggested with separate nets in Agent-57
        return np.add(alpha*ext_values[0],beta*int_values[0])

    # Given a state, uses the CNN as a policy to choose an action
    def choose_action(self,state,frame,alpha,beta):
        # do epsilon soft
        if np.random.uniform(0,1) <= self.get_epsilon(frame):
            return np.random.randint(self.A_SIZE)
        else:
            q_options = self.combine_q(state,alpha,beta)
            
            return np.argmax(q_options)

    def update_target_model(self):
        self.ext_target_model.set_weights(self.ext_local_model.get_weights())
        self.int_target_model.set_weights(self.int_local_model.get_weights())

    def save_state(self,directory):
        int_filename = os.path.join(directory, "int_weights.hdf5")
        self.int_local_model.save_weights(int_filename)
        print(f"Agent network weights were saved in: {int_filename}")

        ext_filename = os.path.join(directory, "ext_weights.hdf5")
        self.ext_local_model.save_weights(ext_filename)
        print(f"Agent network weights were saved in: {ext_filename}")

    def load_state(self,directory):
        int_filename = os.path.join(directory, "int_weights.hdf5")
        ext_filename = os.path.join(directory, "ext_weights.hdf5")
        self.int_local_model.load_weights(int_filename)
        self.ext_local_model.load_weights(ext_filename)
        self.update_target_model()
        print(f"Agent network weights were loaded from: {int_filename} and {ext_filename}")
