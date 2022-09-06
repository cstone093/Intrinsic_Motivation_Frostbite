# hyp from https://github.com/bhctsntrk/OpenAIPong-DQN/blob/master/OpenAIPong_DQN.ipynb
frostbite = {
    "DO_CROP":True,
    "DO_RESCALE":True,
    "RESCALE_DIMS":(84,84),
    "EVALUATION":False,
    "CROP":(42,25,15,0),
    "SEED":0,

    "BUFFER_SIZE":1_000_000,
    "BATCH_SIZE":32,

    "REWARD_BUFFER_SIZE":100_000,
    "ENV":"ALE/Frostbite-v5",
    "FRAMESKIP":True,
    "FRAMESKIP_COUNT":5,
    "GAMMA":0.99,

    "REPLAY_FREQ":4,
    "UPDATE_TARGET":1_000,

    "EPS_FRAMES_INIT":10_000,
    "EPS_FRAMES_INTER":200_000, 
    "EPS_FRAMES_FINAL":1_000_000,

    "EPS_STEPS_INIT":1,
    "EPS_STEPS_INTER":0.1,
    "EPS_STEPS_FINAL":0.01,
    "INIT_LEARNING_RATE":0.0001,

    "STACK_SIZE": 4,
    "MAX_EP_ACTIONS":5_000, 
    
    "RENDER_EVERY_N": 100,  # Render gif every N episodes
    "UPDATE_EVERY_N": 5,  # Print update every N episodes
    "SAVE_EVERY_N": 250, # Save every N episodes
}
