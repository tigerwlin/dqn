import argparse
import logging
import os
import sys
from DQN import DQN
from doubleDQN import DoubleDQN
import copy
import random
import matplotlib.pyplot as plt
import gym
import tempfile
import gym_ple

if __name__ == "__main__":



    random.seed(1)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Catcher-v0', help='Select the environment to run')
    parser.add_argument('--load', help='load network from file')

    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(asctime)s]:%(levelname)s:%(name)s:%(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    #
    # # You can set the level to logging.DEBUG or logging.WARN if you
    # # want to change the amount of output.
    logger.setLevel(logging.DEBUG)

    if args.load is not None:
        outdir = args.load
        logger.info("load network from: " + outdir)
    else:
        outdir = tempfile.mkdtemp()

    logger.info("project dir: " + outdir)

    file_logger = logging.getLogger('file')
    file_logger.handlers = []
    file_logger.propagate = False
    hdlr_2 = logging.FileHandler(outdir+'/record.log')
    hdlr_2.setFormatter(formatter)
    file_logger.addHandler(hdlr_2)
    file_logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    def first_ten(id):
        return id < 10

    # agent = DQN((4,84,84), env.action_space)
    # agent = DQN((4,64,64), env.action_space)
    agent = DoubleDQN((4,64,64), env.action_space)
    # agent = neonDQN((4,64,64), env.action_space)
    if args.load is not None:
        agent.load_network(prefix=outdir+'/network')

    episode_count = 1000000
    max_steps = 5000

    done = False
    steps = 0
    epoch_reward = 0
    episode_num = 0
    for i in range(episode_count):
        state = env.reset()

        episode_reward = [0,0,0]
        episode_steps = 0
        for j in range(max_steps):
        # while not done:
            action = agent.act(state)

            nextState, reward, done, _ = env.step(action)
            episode_reward[0] += reward
            episode_reward[1] += max(reward,0)
            episode_reward[2] += min(reward,0)

            agent.observe(state, action, reward, nextState, done)

            state = copy.deepcopy(nextState)
            episode_steps += 1
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        logger.info("episode steps: " + str(episode_steps) + " reward: " + str(episode_reward))
        steps += episode_steps
        logger.info("total steps: " + str(steps))
        episode_num += 1
        epoch_reward += episode_reward[0]

        if agent.mode == 'train' and (steps-50000) > 25000: # test the model
            agent.save_network(prefix=outdir+'/network')
            logger.info('checkpoint reached, network saved!')
            agent.mode = 'test'
            env.monitor.start(outdir, force=True, seed=0, video_callable=first_ten)
            logger.info("[nums episodes, avg. reward: ] " + str(episode_num) + ", " + str(epoch_reward*1.0/episode_num))
            logger.info("SWITCH TO TESTING MODE!")
            steps -= 25000
            episode_num = 0
            epoch_reward = 0
        if agent.mode == 'test' and (steps-50000) > 12500: # test the model
            # env.monitor.configure(video_callable=lambda count: False)
            logger.info("video saved : " + outdir)
            env.monitor.close()
            agent.mode = 'train'
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

