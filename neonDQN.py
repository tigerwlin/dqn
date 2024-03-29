import numpy as np
import mxnet as mx
import cv2
import random
import matplotlib.pyplot as plt
import copy

#=====================================================
import argparse
# from deepqnetwork import DeepQNetwork
from deepqnetwork_mxnet import DeepQNetwork
from replay_memory import ReplayMemory
#=====================================================


class neonDQN(object):

    def __init__(self, input_shape, action_space):
        self._debug = 0
        self.mode = 'train'
        self.input_shape = input_shape
        self.action_space = action_space
        self.prev_action = action_space.sample()
        self.action_space_size = action_space.n
        self.steps = 0
        self.prelearning_steps = 50000 #50000
        self.total_steps = 10000 #1000000
        self.history_length = input_shape[0]
        self.history_step = 0
        self.observation_buffer = np.zeros(input_shape)
        # self.prev_state = np.zeros(input_shape[1:])
        # learning related
        self.learning_rate = 0.00025
        self.rmsprop_gamma2 = 1
        # experience replay related
        self.memoryIdx = 0
        self.memoryFillCount = 0
        self.memoryLimit = 50000 #1000000
        self.sampleSize = 32

        self.states = np.zeros((self.memoryLimit,) + self.input_shape[1:], dtype='uint8')
        self.actions = np.zeros((self.memoryLimit, ), dtype='uint8')
        self.rewards = np.zeros((self.memoryLimit, ))
        self.nextStates = np.zeros_like(self.states, dtype='uint8')
        self.dones = np.zeros_like(self.actions, dtype='bool')
        # target network update related
        self.targetNetC = 4 #10000
        # Q learning related
        self.gamma = 0.99

        #build Q-learning networks
        print "building network......"
        self.args = self.generate_parameter()
        self.net = self.build_network(self.args)
        self.mem = ReplayMemory(self.memoryLimit, self.args)

        np.set_printoptions(precision=4, suppress=True)

    def act(self, observation):
        observation = self.preprocess_state(observation)
        self.observation_buffer[:-1,...] = self.observation_buffer[1:,...]
        self.observation_buffer[-1,...] = observation

        if self.mode == 'train':
            epsilon = max(0.1, 1-max(self.steps - self.prelearning_steps, 0)/self.total_steps)
        elif self.mode == 'test':
            epsilon = .05
        else:
            assert False

        action = self.choose_action(self.observation_buffer, epsilon)
        return action

    def observe(self, state, action, reward, nextState, done):
        if self.mode == 'test':
            return

        state = self.preprocess_state(state)
        # self.prev_state = state
        nextState = self.preprocess_state(nextState)
        # self.prev_state = nextState

        self.steps += 1
        # ==========================================================
        # plt.figure(2)
        # plt.subplot(3, 1, 1)
        # plt.imshow(state)
        # plt.title("action: " + str(action) + "reward: " + str(reward)
        #           + "done: " + str(done))
        # plt.colorbar()
        # plt.subplot(3, 1, 2)
        # plt.imshow(nextState)
        # plt.subplot(3, 1, 3)
        # plt.imshow(nextState.astype('int16') - state)
        # plt.colorbar()
        # plt.show()
        # ==========================================================
        self.putInMemory(state, action, reward, nextState, done)
        # ==========================================================
        self.mem.add(action, reward, nextState, done)
        # ==========================================================

        if self.steps - self.prelearning_steps > 0: # learning starts

            # state, action, reward, nextState, done = self.sampleFromMemory()
            # ==========================================================
            state, action, reward, nextState, done = self.mem.getMinibatch()
            # ==========================================================
            self.train(state, action, reward, nextState, done)

    def preprocess_state(self, state):
        # state_resize = imresize(state, (84, 84, 3))
        # state_resize_gray = np.mean(state_resize, axis=2)
        # max_state = np.maximum(prev_state, state_resize_gray)
        # return max_state.astype('uint8')
        state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), self.input_shape[1:])
        return state

    def putInMemory(self, state, action, reward, nextState, done):
        memoryIdx = self.memoryIdx
        self.states[memoryIdx, ...] = state
        self.actions[memoryIdx, ...] = action
        self.rewards[memoryIdx, ...] = reward
        self.nextStates[memoryIdx, ...] = nextState
        self.dones[memoryIdx, ...] = done

        self.memoryIdx += 1
        self.memoryFillCount = max(self.memoryFillCount, self.memoryIdx)
        assert self.memoryFillCount <= self.memoryLimit
        self.memoryIdx = self.memoryIdx % self.memoryLimit

    def sampleFromMemory(self):
        # sampleIdx = np.random.permutation(self.memoryLimit)
        # sampleIdx = sampleIdx[:self.sampleSize]
        #
        # state = np.zeros((self.sampleSize,) + self.states.shape[1:])
        # action = np.zeros((self.sampleSize,) + self.actions.shape[1:], dtype='int')
        # reward = np.zeros((self.sampleSize,) + self.rewards.shape[1:])
        # nextState = np.zeros((self.sampleSize,) + self.nextStates.shape[1:])
        # done = np.zeros((self.sampleSize,) + self.dones.shape[1:], dtype='int')
        #
        # for i in xrange(self.sampleSize):
        #     state[i] = self.states[sampleIdx[i]]
        #     action[i] = self.actions[sampleIdx[i]]
        #     reward[i] = self.rewards[sampleIdx[i]]
        #     nextState[i] = self.nextStates[sampleIdx[i]]
        #     done[i] = self.dones[sampleIdx[i]]
        #
        # return state, action, reward, nextState, done
    #==================================================================================================
        state = np.zeros((self.sampleSize, self.history_length) + self.states.shape[1:], dtype='uint8')
        nextState = np.zeros((self.sampleSize, self.history_length) + self.nextStates.shape[1:], dtype='uint8')
        indexes = []
        while len(indexes) < self.sampleSize:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length-1, self.memoryFillCount-1)
                # if wraps over current pointer, then get new one
                if index >= self.memoryIdx and index - (self.history_length - 1) < self.memoryIdx:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.dones[(index - self.history_length + 1):index].any():
                    continue
                # if (self.rewards[(index - self.history_length + 1):index] != 0).any():
                #     continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            assert index >= self.history_length-1
            assert index <= self.memoryLimit-1
            state[len(indexes), ...] = self.states[(index - (self.history_length - 1)):(index + 1), ...]
            nextState[len(indexes), ...] = self.nextStates[(index - (self.history_length - 1)):(index + 1), ...]
            indexes.append(index)

        # copy actions, rewards and terminals with direct slicing
        action = self.actions[indexes]
        reward = self.rewards[indexes]
        done = self.dones[indexes]
        return state, action, reward, nextState, done


    def build_network(self, args):
        net = DeepQNetwork(self.action_space_size, args)
        return net

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.action_space.sample()
        else:
            return self.greedy(state)

    def greedy(self, state):
        # predict the Q values at current state
        state = state[np.newaxis,:]
        #replicate by batch_size
        state = np.tile(state, (self.sampleSize,1,1,1))

        # ======================================================
        q = self.net.predict(state)
        #======================================================
        # q = self._network_forward(self.network, state)
        # ======================================================

        q = q[0,:]
        # return the index of maximum Q value
        return np.argmax(q)

    def _network_forward(self, net, state):
        assert state.shape[0] == self.sampleSize
        assert state.shape[1] == self.input_shape[0]

        state = state / 255.0
        arg_arrays = net.arg_dict
        train_iter = mx.io.NDArrayIter(data=state, batch_size=state.shape[0])
        data = arg_arrays[train_iter.provide_data[0][0]]

        q = []
        for batch in train_iter:
            # Copy data to executor input. Note the [:].
            data[:] = batch.data[0]

            self.network.forward(is_train=False)

            q = self.network.outputs[0]

        return q.asnumpy()

    def train(self, state, action, reward, nextState, done):
        epoch = 0
        minibatch = state, action, reward, nextState, done
        self.net.train(minibatch, epoch)
        # reward = np.clip(reward, -1, 1)
        #
        #
        # future_Qvalue = self._network_forward(self.targetNetwork, nextState)
        # future_reward = np.max(future_Qvalue, axis=1)
        # future_reward = future_reward[:, np.newaxis]
        #
        # nonzero_reward_list = np.nonzero(reward)
        # # reward += (1-done)*self.gamma*future_reward
        # reward += (1-abs(reward))*self.gamma*future_reward
        #
        # target_reward = self._network_forward(self.network, state)
        # old_target_reward = copy.deepcopy(target_reward)
        # for i in xrange(self.sampleSize):
        #     # target_reward[i][action[i]] = reward[i]
        #     # clip error to [-1, 1], Mnih 2015 Nature
        #     target_reward[i][action[i]] = max(min(reward[i], target_reward[i][action[i]]+1), target_reward[i][action[i]]-1)
        #
        # #=======================================================================
        # if self._debug:
        #     print "reward:", reward.transpose()
        #     print "future_reward:", future_reward.transpose()
        #     print "action:", action.transpose()
        #     print "done: ", done.transpose()
        #     figure_id = 0
        #     for batch_i in nonzero_reward_list[0]:
        #         if 1: #reward[batch_i, ...] != 0:
        #             figure_id += 1
        #             plt.figure(figure_id)
        #             for plot_i in range(0, self.history_length):
        #                 plt.subplot(3, self.history_length, plot_i + 1)
        #                 plt.imshow(state[batch_i, plot_i, ...])
        #                 plt.title("action: " + str(action[batch_i, ...]) + "reward: " + str(reward[batch_i, ...])
        #                           + "done: " + str(done[batch_i, ...]))
        #                 plt.colorbar()
        #
        #                 plt.subplot(3, self.history_length, plot_i + 1 + self.history_length)
        #                 plt.imshow(nextState[batch_i, plot_i, ...])
        #
        #                 plt.subplot(3, self.history_length, plot_i + 1 + self.history_length * 2)
        #                 plt.imshow(nextState[batch_i, plot_i, ...].astype('int16') - state[batch_i, plot_i, ...])
        #                 if plot_i == 0:
        #                     plt.title("reward: " + str(reward[batch_i, ...])
        #                           + " target reward: " + str(target_reward[batch_i, ...])
        #                           + " old reward: " + str(old_target_reward[batch_i, ...]))
        #                 plt.colorbar()
        #
        #     plt.show()
        #     # raw_input()
        # #=======================================================================
        #
        # train_data = state / 255.0
        # train_label = target_reward
        #
        #
        # # First we get handle to input arrays
        # arg_arrays = self.network.arg_dict
        # batch_size = self.sampleSize
        # train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=False)
        # # val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=batch_size)
        # data = arg_arrays[train_iter.provide_data[0][0]]
        # label = arg_arrays[train_iter.provide_label[0][0]]
        #
        # # opt = mx.optimizer.RMSProp(
        # #     learning_rate= self.learning_rate,
        # #     gamma2 = self.rmsprop_gamma2)
        #
        # opt = mx.optimizer.Adam(
        #     learning_rate=self.learning_rate)
        #
        # updater = mx.optimizer.get_updater(opt)
        #
        # # Finally we need a metric to print out training progress
        # metric = mx.metric.MSE()
        #
        # # Training loop begines
        # train_iter.reset()
        # metric.reset()
        #
        # for batch in train_iter:
        #     # Copy data to executor input. Note the [:].
        #     data[:] = batch.data[0]
        #     label[:] = batch.label[0]
        #
        #     # Forward
        #     self.network.forward(is_train=True)
        #
        #     # You perform operations on exe.outputs here if you need to.
        #     # For example, you can stack a CRF on top of a neural network.
        #
        #     # Backward
        #     self.network.backward()
        #
        #     # Update
        #     for i, pair in enumerate(zip(self.network.arg_arrays, self.network.grad_arrays)):
        #         weight, grad = pair
        #         updater(i, grad, weight)
        #     metric.update(batch.label, self.network.outputs)
        #
        #     if self.steps % 1000 == 0:
        #         print 'steps:', self.steps, 'metric:', metric.get()
        #         print 'network.outputs:', self.network.outputs[0].asnumpy()
        #         print 'label:', batch.label[0].asnumpy()
        #         # np.set_printoptions(precision=4)
        #         print 'delta: ', (batch.label[0].asnumpy() - self.network.outputs[0].asnumpy())
        # # t = 0
        # # metric.reset()
        # # for batch in val_iter:
        # #     # Copy data to executor input. Note the [:].
        # #     data[:] = batch.data[0]
        # #     label[:] = batch.label[0]
        # #
        # #     # Forward
        # #     self.network.forward(is_train=False)
        # #     metric.update(batch.label, self.network.outputs)
        # #     t += 1
        # #     if t % 50 == 0:
        # #         print 'epoch:', epoch, 'test iter:', t, 'metric:', metric.get()
        #
        # #========================================================================
        # #sync target-network with network as mentioned in Mnih et al. Nature 2015
        if self.steps % self.targetNetC == 0:
            self.net.update_target_network()
        #     self.targetNetwork.copy_params_from(self.network.arg_dict, self.network.aux_dict)


    # Basic Conv + BN + ReLU factory
    def ConvFactory(self, data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
        # there is an optional parameter ```wrokshpace``` may influece convolution performance
        # default, the workspace is set to 256(MB)
        # you may set larger value, but convolution layer only requires its needed but not exactly
        # MXNet will handle reuse of workspace without parallelism conflict
        conv = mx.symbol.Convolution(data=data, workspace=256,
                                     num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
        # bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = conv, act_type=act_type)
        return act

    def generate_parameter(self):

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")

        parser = argparse.ArgumentParser()

        envarg = parser.add_argument_group('Environment')
        envarg.add_argument("--game", default="Catcher-v0", help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym.")
        envarg.add_argument("--environment", choices=["ale", "gym"], default="ale",
                            help="Whether to train agent using ALE or OpenAI Gym.")
        envarg.add_argument("--display_screen", type=str2bool, default=False,
                            help="Display game screen during training and testing.")
        # envarg.add_argument("--sound", type=str2bool, default=False, help="Play (or record) sound.")
        envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
        envarg.add_argument("--repeat_action_probability", type=float, default=0,
                            help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
        envarg.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True,
                            help="Use minimal action set.")
        envarg.add_argument("--color_averaging", type=str2bool, default=True,
                            help="Perform color averaging with previous frame.")
        envarg.add_argument("--screen_width", type=int, default=64, help="Screen width after resize.")
        envarg.add_argument("--screen_height", type=int, default=64, help="Screen height after resize.")
        envarg.add_argument("--record_screen_path", default="./",
                            help="Record game screens under this path. Subfolder for each game is created.")
        envarg.add_argument("--record_sound_filename", default="./", help="Record game sound in this file.")

        memarg = parser.add_argument_group('Replay memory')
        memarg.add_argument("--replay_size", type=int, default=50000, help="Maximum size of replay memory.")
        memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

        netarg = parser.add_argument_group('Deep Q-learning network')
        netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
        netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
        netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
        netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop',
                            help='Network optimization algorithm.')
        netarg.add_argument("--decay_rate", type=float, default=0.95,
                            help="Decay rate for RMSProp and Adadelta algorithms.")
        netarg.add_argument("--clip_error", type=float, default=1,
                            help="Clip error term in update between this number and its negative.")
        netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
        netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
        netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")

        # netarg.add_argument("--rescale_r", type=str2bool, help="Rescale rewards.")
        # missing: bufferSize=512,valid_size=500,min_reward=-1,max_reward=1

        neonarg = parser.add_argument_group('Neon')
        neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
        neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
        neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32',
                             help='default floating point precision for backend [f64 for cpu only]')
        neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False,
                             help='use stochastic rounding [will round to BITS number of bits if specified]')

        antarg = parser.add_argument_group('Agent')
        antarg.add_argument("--exploration_rate_start", type=float, default=1,
                            help="Exploration rate at the beginning of decay.")
        antarg.add_argument("--exploration_rate_end", type=float, default=0.1,
                            help="Exploration rate at the end of decay.")
        antarg.add_argument("--exploration_decay_steps", type=float, default=10000,
                            help="How many steps to decay the exploration rate.")
        antarg.add_argument("--exploration_rate_test", type=float, default=0.05,
                            help="Exploration rate used during testing.")
        antarg.add_argument("--train_frequency", type=int, default=4,
                            help="Perform training after this many game steps.")
        antarg.add_argument("--train_repeat", type=int, default=1,
                            help="Number of times to sample minibatch during training.")
        antarg.add_argument("--target_steps", type=int, default=4,
                            help="Copy main network to target network after this many game steps.")
        antarg.add_argument("--random_starts", type=int, default=30,
                            help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

        nvisarg = parser.add_argument_group('Visualization')
        nvisarg.add_argument("--visualization_filters", type=int, default=4,
                             help="Number of filters to visualize from each convolutional layer.")
        nvisarg.add_argument("--visualization_file", default="tmp", help="Write layer visualization to this file.")

        mainarg = parser.add_argument_group('Main loop')
        mainarg.add_argument("--random_steps", type=int, default=50000,
                             help="Populate replay memory with random steps before starting learning.")
        mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
        mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
        mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
        mainarg.add_argument("--start_epoch", type=int, default=0,
                             help="Start from this epoch, affects exploration rate and names of saved snapshots.")
        mainarg.add_argument("--play_games", type=int, default=0,
                             help="How many games to play, suppresses training and testing.")
        mainarg.add_argument("--load_weights", help="Load network from file.")
        mainarg.add_argument("--save_weights_prefix",
                             help="Save network to given file. Epoch and extension will be appended.")
        mainarg.add_argument("--csv_file", help="Write training progress to this file.")

        comarg = parser.add_argument_group('Common')
        comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
        comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                            help="Log level.")
        args = parser.parse_args()
        return args
