import argparse
import logging
import os
import sys
from DQN import DQN
from doubleDQN import DoubleDQN
from duel_network import DuelNetwork
import copy
import random
import matplotlib.pyplot as plt
import gym
import gym_ple
import tempfile


def get_parameters():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Catcher-v0', help='Select the environment to run')
    parser.add_argument('--model', choices=['DQN', 'doubleDQN', 'duelNetwork'], default='DQN', help='choose network models')
    parser.add_argument('--load', help='load network from file')
    parser.add_argument("--screen_width", type=int, default=64, help="Screen width after resize.")
    parser.add_argument("--screen_height", type=int, default=64, help="Screen height after resize.")
    parser.add_argument('--prelearning_steps', type=int, default=50000, help='random steps at the beginning to fill the replay memory')
    parser.add_argument('--total_steps', type=int, default=10000,
                        help='steps taken to reduce the exploration rate (epsilon)')
    parser.add_argument('--learning_rate', type=float, default=0.00025,
                        help='learning rate')
    parser.add_argument('--epsilon_training_bound', type=float, default=0.1,
                        help='minimum exploration rate in training')
    parser.add_argument('--epsilon_testing', type=float, default=0.05,
                        help='exploration rate in testing')
    parser.add_argument('--memoryLimit', type=int, default=50000,
                        help='size of the replay memory')
    parser.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")
    parser.add_argument('--sampleSize', type=int, default=32,
                        help='size of the minibatch in training')
    parser.add_argument('--targetNetC', type=int, default=4,
                        help='update interval of target network')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount rate of future reward')
    parser.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop',
                        help='Network optimization algorithm.')
    parser.add_argument("--decay_rate", type=float, default=0.95,
                        help="Decay rate for RMSProp and Adadelta algorithms.")
    parser.add_argument("--train_steps", type=int, default=25000, help="How many training steps per epoch.")
    parser.add_argument("--test_steps", type=int, default=12500, help="How many testing steps after each epoch.")
    parser.add_argument("--record_videos", type=int, default=0, help="Whether record videos during testing.")
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
    parser.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_parameters()

    random.seed(1)


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
    logger.setLevel(logging.INFO)

    if args.load is not None:
        outdir = args.load
        logger.info("load network from: " + outdir)
    else:
        outdir = tempfile.mkdtemp(dir='/mnt/scratch/wulin/tmp', suffix='_'+args.env_id)

    logger.info("project dir: " + outdir)

    file_logger = logging.getLogger('file')
    file_logger.handlers = []
    file_logger.propagate = False
    hdlr_2 = logging.FileHandler(outdir+'/record.log')
    hdlr_2.setFormatter(formatter)
    file_logger.addHandler(hdlr_2)
    file_logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    def first_ten(id):
        return id < 10

    if args.model == 'DQN':
        agent = DQN(args, env.action_space)
    elif args.model == 'doubleDQN':
        agent = DoubleDQN(args, env.action_space)
    elif args.model == 'duelNetwork':
        agent = DuelNetwork(args, env.action_space)
    else:
        assert False
    # agent = neonDQN((4,64,64), env.action_space)
    if args.load is not None:
        agent.load_network(prefix=outdir+'/network')

    # episode_count = 1000000
    epoch_count = 2000
    max_steps = 5000

    done = False
    steps = 0
    prelearning_steps = args.prelearning_steps

    for i in range(epoch_count):    # start an training/testing epoch
        epoch_steps = min(0, -prelearning_steps)
        prelearning_steps = 0
        epoch_reward = 0
        episode_num = 0
        while 1:    # start a new episode
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

                if file_logger.isEnabledFor(logging.DEBUG):
                    file_logger.debug("model: " + str(args.model) + " current step score: " + str(episode_reward[0]))
                if done:
                    break
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

            logger.info("episode steps: " + str(episode_steps) + " reward: " + str(episode_reward))
            steps += episode_steps
            epoch_steps += episode_steps
            logger.info("total steps: " + str(steps))
            episode_num += 1
            epoch_reward += episode_reward[0]

            if agent.mode == 'train' and epoch_steps > args.train_steps: # test the model
                agent.save_network(prefix=outdir+'/network')
                logger.info('checkpoint reached, network saved!')
                agent.mode = 'test'
                if args.record_videos:
                    env.monitor.start(outdir, force=True, seed=0, video_callable=first_ten)
                else:
                    env.monitor.start(outdir, force=True, seed=0, video_callable=False)
                assert episode_num != 0
                logger.info("[nums episodes, avg. reward: ] " + str(episode_num) + ", " + str(epoch_reward*1.0/episode_num))
                logger.info("SWITCH TO TESTING MODE!")
                break   # exit this epoch
            if agent.mode == 'test' and epoch_steps > args.test_steps:
                # env.monitor.configure(video_callable=lambda count: False)
                logger.info("video saved : " + outdir)
                env.monitor.close()
                agent.mode = 'train'
                assert episode_num != 0
                logger.info("[nums episodes, avg. reward: ] " + str(episode_num) + ", " + str(epoch_reward*1.0/episode_num))
                logger.info("SWITCH TO TRAINING MODE!")
                break   # exit this epoch

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    # agent = RandomAgent(env.action_space)
    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran your program. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)

