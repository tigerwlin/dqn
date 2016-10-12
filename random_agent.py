import argparse
import logging
import os
import sys
from DQN import DQN
# from neonDQN import neonDQN
import numpy as np
import copy
import random
from scipy.misc import imresize
import matplotlib.pyplot as plt
import gym
import tempfile
import gym_ple
# # The world's simplest agent!
# class RandomAgent(object):
#     def __init__(self, action_space):
#         self.action_space = action_space
#
#     def act(self, observation, reward, done):
#         return self.action_space.sample()


if __name__ == "__main__":
    random.seed(1)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Catcher-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.handlers = []
    # formatter = logging.Formatter('[%(asctime)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s]:%(levelname)s:%(name)s:%(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    #
    # # You can set the level to logging.DEBUG or logging.WARN if you
    # # want to change the amount of output.
    logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    def first_ten(id):
        return id < 10
    # outdir = '/tmp/random-agent-results'
    outdir = tempfile.mkdtemp()
    # env.monitor.start(outdir, force=True, seed=0, video_callable=lambda count: False)

    # input_shape = env.observation_space.shape
    # agent = DQN((input_shape[-1],)+input_shape[:-1], env.action_space)
    # agent = DQN((4,84,84), env.action_space)
    agent = DQN((4,64,64), env.action_space)
    # agent = neonDQN((4,64,64), env.action_space)

    episode_count = 1000000
    max_steps = 5000

    done = False
    steps = 0
    epoch_reward = 0
    episode_num = 0
    for i in range(episode_count):
        state = env.reset()
        #state = preprocess_state(state, state, state, state)
        # state = preprocess_state(state)

        episode_reward = [0,0,0]
        episode_steps = 0
        for j in range(max_steps):
        # while not done:
            action = agent.act(state)
            # action_ghost = agent_ghost.act(state)
            # action = int(raw_input('enter action: '))

            nextState, reward, done, _ = env.step(action)
            episode_reward[0] += reward
            episode_reward[1] += max(reward,0)
            episode_reward[2] += min(reward,0)
            # nextState = preprocess_state(nextState)

            agent.observe(state, action, reward, nextState, done)
            # agent_ghost.observe(state, action, reward, nextState, done)

            state = copy.deepcopy(nextState)
            episode_steps += 1
            # logger.debug("episode step increase by 1 to : " + str(steps+episode_steps))
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        # print "episode steps: ", episode_steps, "reward: ", episode_reward
        logger.info("episode steps: " + str(episode_steps) + " reward: " + str(episode_reward))
        steps += episode_steps
        logger.info("total steps: " + str(steps))
        episode_num += 1
        epoch_reward += episode_reward[0]

        if agent.mode == 'train' and (steps-50000) > 25000: # test the model
            agent.save_network(outdir+'/network')
            logger.info('checkpoint reached, network saved!')
            agent.mode = 'test'
            # env.monitor.configure(video_callable=lambda count: count % 1000 == 0)
            env.monitor.start(outdir, force=True, seed=0, video_callable=first_ten)
            # print "[nums episodes, avg. reward: ] ", epoch_num, epoch_reward*1.0/epoch_num
            # print "SWITCH TO TESTING MODE!"
            logger.info("[nums episodes, avg. reward: ] " + str(episode_num) + ", " + str(epoch_reward*1.0/episode_num))
            logger.info("SWITCH TO TESTING MODE!")
            steps -= 25000
            episode_num = 0
            epoch_reward = 0
        if agent.mode == 'test' and (steps-50000) > 12500: # test the model
            # env.monitor.configure(video_callable=lambda count: False)
            env.monitor.close()
            agent.mode = 'train'
            # print "[nums episodes, avg. reward: ] ", epoch_num, epoch_reward*1.0/epoch_num
            # print "SWITCH TO TRAINING MODE!"
            logger.info("[nums episodes, avg. reward: ] " + str(episode_num) + ", " + str(epoch_reward*1.0/episode_num))
            logger.info("SWITCH TO TRAINING MODE!")
            steps -= 12500
            episode_num = 0
            epoch_reward = 0


    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    # agent = RandomAgent(env.action_space)
    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)

