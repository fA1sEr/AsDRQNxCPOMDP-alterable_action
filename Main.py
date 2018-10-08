import itertools as it
import os
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
from vizdoom import *
from Agent import Agent
from GameSimulator.hallway_GameSimulator import GameSimulator

# to choose gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

FRAME_REPEAT = 1 # How many frames 1 action should be repeated
UPDATE_FREQUENCY = 4
COPY_FREQUENCY = 1000

STATE_NUM = 21
ACTION_LENGTH = 5 # change two place [1]
STATE_LENGTH = STATE_NUM + ACTION_LENGTH
BATCH_SIZE = 32 # Batch size for experience replay
LEARNING_RATE = 0.001 # Learning rate of model
GAMMA = 0.95 # Discount factor

MEMORY_CAP = 10000000 # Amount of samples to store in memory

EPSILON_MAX = 1 # Max exploration rate
EPSILON_MIN = 0.1 # Min exploration rate
EPSILON_DECAY_STEPS = 3e5 # How many steps to decay from max exploration to min exploration

RANDOM_WANDER_STEPS = 200000 # How many steps to be sampled randomly before training starts

TRACE_LENGTH = 8 # How many traces are used for network updates
HIDDEN_SIZE = 768 # Size of the third convolutional layer when flattened

EPOCHS = 200 # Epochs for training (1 epoch = 200 training Games and 10 test episodes)
GAMES_PER_EPOCH = 100 # How actions to be taken per epoch
EPISODES_TO_TEST = 100 # How many test episodes to be run per epoch for logging performance
FINAL_TO_TEST = 1000
EPISODE_TO_WATCH = 10 # How many episodes to watch after training is complete

TAU = 0.99 # How much the target network should be updated towards the online network at each update

LOAD_MODEL = False # Load a saved model?
SAVE_MODEL = True # Save a model while training?
SKIP_LEARNING = False # Skip training completely and just watch?

max_model_savefile = "train_data/max_model/max_model.ckpt"
model_savefile = "train_data/model.ckpt" # Name and path of the model
reward_savefile = "train_data/Rewards.txt"

##########################################

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def saveScore(score, suc_rate, avg_step):
    my_file = open(reward_savefile, 'a')  # Name and path of the reward text file
    my_file.write("avg reward:%s(%s) suc rate:%s avg step:%s\n" % (score.mean(), score.std(), suc_rate, avg_step))
    my_file.close()

###########################################

game = GameSimulator()
game.initialize()

ACTION_COUNT = game.get_action_size()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
SESSION = tf.Session(config=config)

if LOAD_MODEL:
    EPSILON_MAX = 0.25 # restart after 20+ epoch

agent = Agent(memory_cap = MEMORY_CAP, batch_size = BATCH_SIZE, state_length = STATE_LENGTH, action_count = ACTION_COUNT,
            session = SESSION, lr = LEARNING_RATE, gamma = GAMMA, epsilon_min = EPSILON_MIN, trace_length=TRACE_LENGTH,
            epsilon_decay_steps = EPSILON_DECAY_STEPS, epsilon_max=EPSILON_MAX, hidden_size=HIDDEN_SIZE)

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, TAU)

if LOAD_MODEL:
    print("Loading model from: ", model_savefile)
    saver.restore(SESSION, model_savefile)
else:
    init = tf.global_variables_initializer()
    SESSION.run(init)

##########################################

if not SKIP_LEARNING:
    time_start = time()
    print("\nFilling out replay memory")
    updateTarget(targetOps, SESSION)

    agent.reset_cell_state()
    state = game.get_state()
    for _ in range(RANDOM_WANDER_STEPS):
        if not LOAD_MODEL:
            action = agent.random_action()
        else:
            action = agent.act(state)
        img_state, reward, done = game.make_action(action)
        if done!=1:
            state_new = img_state
        else:
            state_new = None

        agent.add_transition(state, action, reward, state_new, done, is_init=True)
        state = state_new

        if done!=0:
            game.reset()
            agent.reset_cell_state()
            state = game.get_state()

    max_avgR = -10000.0

    for epoch in range(EPOCHS):
        print("\n\nEpoch %d\n-------" % (epoch + 1))
        print("Training...")

        learning_step = 0
        success_num = 0
        for games_cnt in range(GAMES_PER_EPOCH):
            game.reset()
            agent.reset_cell_state()
            state = game.get_state()
            while True:
                learning_step += 1
                action = agent.act(state)
                img_state, reward, done = game.make_action(action)
                if done!=1:
                    state_new = img_state
                else:
                    state_new = None
                agent.add_transition(state, action, reward, state_new, done)
                state = state_new

                if learning_step % UPDATE_FREQUENCY == 0:
                    agent.learn_from_memory()
                if learning_step % COPY_FREQUENCY == 0:
                    updateTarget(targetOps, SESSION)

                if done!=0:
                    print("Epoch %d Train Game %d get %.1f" % (epoch, games_cnt, game.get_total_reward()))
                    if game.get_total_reward()>0:
                        success_num += 1
                    break
            if SAVE_MODEL and games_cnt % 50 == 0:
                saver.save(SESSION, model_savefile)
                #print("Saving the network weigths to:", model_savefile)

        print('train success rate:',success_num/GAMES_PER_EPOCH)
        print('\nTesting...')

        success_num = 0
        success_total_step = 0
        test_scores = []
        if epoch==EPOCHS-1:
            test_game_num = FINAL_TO_TEST
        else:
            test_game_num = EPISODES_TO_TEST
        for test_step in range(test_game_num):
            game.reset()
            agent.reset_cell_state()
            total_step = 0
            while game.is_terminated()==0:
                state = game.get_state()
                action = agent.act(state, train=False)
                game.make_action(action, train=False)
                total_step += 1
            test_scores.append(game.get_total_reward())
            if game.get_total_reward()>0:
                success_num += 1
                success_total_step += total_step

        test_scores = np.array(test_scores)
        print('test success rate:',success_num/test_game_num)
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()),
              "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

        if SAVE_MODEL:
            if success_num==0:
                avg_step = 0
            else:
                avg_step = success_total_step/success_num
            saveScore(test_scores, success_num/test_game_num, avg_step)
            saver.save(SESSION, model_savefile)
            print("Saving the network weigths to:", model_savefile)
            if test_scores.mean() > max_avgR:
                max_avgR = test_scores.mean()
                saver.save(SESSION, max_model_savefile)

        print("Total ellapsed time: %.2f minutes" % ((time() - time_start) / 60.0))