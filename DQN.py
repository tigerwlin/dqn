import numpy as np
import mxnet as mx
import cv2
import random
import json

import logging
logger = logging.getLogger(__name__)
file_logger = logging.getLogger('file')


class DQN(object):

    def __init__(self, args, action_space):
        self._debug = 0
        self.mode = 'train'
        self.model = args.model
        self.input_shape = (args.history_length, args.screen_width, args.screen_height)
        self.action_space = action_space
        # self.prev_action = action_space.sample()
        self.action_space_size = action_space.n
        self.steps = 0
        self.prelearning_steps = args.prelearning_steps #50000
        self.total_steps = args.total_steps #1000000
        self.history_length = args.history_length
        # self.history_step = 0
        self.observation_buffer = np.zeros(self.input_shape)
        # self.prev_state = np.zeros(input_shape[1:])
        # learning related
        self.learning_rate = args.learning_rate # 0.00025
        self.optimizer = args.optimizer
        self.decay_rate = args.decay_rate
        self.rmsprop_gamma2 = 0.
        self.epsilon_training_bound = args.epsilon_training_bound # 0.1
        self.epsilon_testing = args.epsilon_testing # 0.05
        # experience replay related
        self.memoryIdx = 0
        self.memoryFillCount = 0
        self.memoryLimit = args.memoryLimit # 50000 #1000000
        self.sampleSize = args.sampleSize # 32

        # self.states = np.zeros((self.memoryLimit,) + self.input_shape[1:], dtype='uint8')
        self.actions = np.zeros((self.memoryLimit,), dtype='uint8')
        self.rewards = np.zeros((self.memoryLimit,))
        self.nextStates = np.zeros((self.memoryLimit,) + self.input_shape[1:], dtype='uint8')
        self.dones = np.zeros_like(self.actions, dtype='bool')
        # target network update related
        self.targetNetC = args.targetNetC # 4 #10000
        # Q learning related
        self.gamma = args.gamma # 0.99

        #build Q-learning networks
        logger.info("building network......")
        self.backend = args.backend
        self.device_id = args.device_id
        if self.backend == 'cpu':
            self.ctx = mx.cpu()
        elif self.backend == 'gpu':
            self.ctx = mx.gpu(self.device_id)
        else:
            assert False

        self.network, self.targetNetwork = self.build_network()

        if self.optimizer == 'rmsprop':
            self.opt = mx.optimizer.RMSProp(
                learning_rate=self.learning_rate, rescale_grad=1.0 / self.sampleSize,
                gamma2=self.rmsprop_gamma2)
        elif self.optimizer == 'adam':
            self.opt = mx.optimizer.Adam(
                learning_rate=self.learning_rate, rescale_grad = 1.0/self.sampleSize)
        else:
            assert False

        self.updater = mx.optimizer.get_updater(self.opt)

        # Finally we need a metric to print out training progress
        self.metric = mx.metric.MSE()


        np.set_printoptions(precision=4, suppress=True)

    def act(self, observation):
        observation = self.preprocess_state(observation)

        self.observation_buffer[:-1,...] = self.observation_buffer[1:,...]
        self.observation_buffer[-1,...] = observation


        if self.mode == 'train':
            if self.prelearning_steps > 0:
                # generate random steps
                epsilon = max(self.epsilon_training_bound, 1-self.steps/self.total_steps)
                self.prelearning_steps -= 1
                if self.prelearning_steps == 0:
                    logger.info("prelearning finish, memory filled, learning starts!")
            else:
                epsilon = max(self.epsilon_training_bound, 1-self.steps/self.total_steps)
                self.steps += 1
        elif self.mode == 'test':
            epsilon = self.epsilon_testing
        else:
            assert False

        action = self.choose_action(self.observation_buffer, epsilon)
        return action

    def observe(self, state, action, reward, nextState, done):
        if self.mode == 'test':
            return

        state = self.preprocess_state(state)
        nextState = self.preprocess_state(nextState)

        self.putInMemory(state, action, reward, nextState, done)

        if self.prelearning_steps <= 0: # learning starts
            # draw a minibatch from the replay memory
            state, action, reward, nextState, done = self.sampleFromMemory()
            # train the network
            self.train(state, action, reward, nextState, done)

    def preprocess_state(self, state):
        state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), self.input_shape[1:])
        return state

    def putInMemory(self, state, action, reward, nextState, done):
        memoryIdx = self.memoryIdx
        # self.states[memoryIdx, ...] = state
        self.actions[memoryIdx, ...] = action
        self.rewards[memoryIdx, ...] = reward
        self.nextStates[memoryIdx, ...] = nextState
        self.dones[memoryIdx, ...] = done

        self.memoryIdx += 1
        self.memoryFillCount = max(self.memoryFillCount, self.memoryIdx)
        assert self.memoryFillCount <= self.memoryLimit
        self.memoryIdx = self.memoryIdx % self.memoryLimit

    def sampleFromMemory(self):

        state = np.zeros((self.sampleSize, self.history_length) + self.nextStates.shape[1:], dtype='uint8')
        nextState = np.zeros((self.sampleSize, self.history_length) + self.nextStates.shape[1:], dtype='uint8')
        indexes = []
        while len(indexes) < self.sampleSize:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                # index = random.randint(self.history_length-1, self.memoryFillCount-1)
                index = random.randint(self.history_length, self.memoryFillCount-1)
                # if wraps over current pointer, then get new one
                if index >= self.memoryIdx and index - (self.history_length - 1) < self.memoryIdx:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.dones[(index - self.history_length + 1):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            assert index >= self.history_length-1
            assert index <= self.memoryLimit-1
            # state[len(indexes), ...] = self.states[(index - (self.history_length - 1)):(index + 1), ...]
            state[len(indexes), ...] = self.nextStates[(index - self.history_length):index, ...]
            nextState[len(indexes), ...] = self.nextStates[(index - (self.history_length - 1)):(index + 1), ...]
            indexes.append(index)

        # copy actions, rewards and terminals with direct slicing
        action = self.actions[indexes]
        reward = self.rewards[indexes]
        done = self.dones[indexes]
        return state, action, reward, nextState, done


    def build_network(self):

        # we can use mx.sym in short of mx.symbol
        data = mx.sym.Variable("data")

        # conv1 = self.ConvFactory(data=data, kernel=(8,8), stride=(4,4), pad=(0,0), num_filter=32, act_type="relu")
        # conv2 = self.ConvFactory(data=conv1, kernel=(4,4), stride=(2,2), pad=(0,0), num_filter=64, act_type="relu")
        # conv3 = self.ConvFactory(data=conv2, kernel=(3,3), stride=(1,1), pad=(0,0), num_filter=64, act_type="relu")
        #====================================================================================================
        conv1 = self.ConvFactory(data=data, kernel=(5, 5), stride=(2, 2), pad=(0, 0), num_filter=32, act_type="relu")
        # conv2 = self.ConvFactory(data=conv1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64, act_type="relu")
        conv3 = self.ConvFactory(data=conv1, kernel=(5, 5), stride=(2, 2), pad=(0, 0), num_filter=32, act_type="relu")

        fc1 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="fc1")
        # bn1 = mx.sym.BatchNorm(data=fc1, name="bn1")
        act1 = mx.sym.Activation(data=fc1, name="act1", act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name="fc2", num_hidden=self.action_space_size)
        # softmax = mx.sym.Softmax(data=fc2, name="softmax")
        softmax = mx.sym.LinearRegressionOutput(data=fc2, name="softmax")

        # visualize the network
        batch_size = self.sampleSize
        data_shape = (batch_size, ) + self.input_shape
        # print data_shape
        logger.info( str(data_shape) )
        mx.viz.plot_network(softmax, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'})
        # print softmax.list_arguments()
        logger.info( str(softmax.list_arguments()) )
        # ==================Binding=====================
        # The symbol we created is only a graph description.
        # To run it, we first need to allocate memory and create an executor by 'binding' it.
        # In order to bind a symbol, we need at least two pieces of information: context and input shapes.
        # Context specifies which device the executor runs on, e.g. cpu, GPU0, GPU1, etc.
        # Input shapes define the executor's input array dimensions.
        # MXNet then run automatic shape inference to determine the dimensions of intermediate and output arrays.

        # We use simple_bind to let MXNet allocate memory for us.
        # You can also allocate memory youself and use bind to pass it to MXNet.
        network = softmax.simple_bind(ctx=self.ctx, data=data_shape)

        #create a dumpy dataset
        train_data = np.zeros((batch_size,) + self.input_shape)
        label_data = np.zeros((batch_size, self.action_space_size))
        train_iter = mx.io.NDArrayIter(data=train_data, label=label_data, batch_size=batch_size, shuffle=True)
        input_shapes = dict(train_iter.provide_data + train_iter.provide_label)
        # print "input_shapes:", input_shapes
        logger.info("input_shapes:" + str(input_shapes))
        # ===============Initialization=================
        # First we get handle to input arrays
        arg_arrays = dict(zip(softmax.list_arguments(), network.arg_arrays))

        # We initialize the weights with Xavier.
        init = mx.init.Xavier(factor_type="in")
        for name, arr in arg_arrays.items():
            if name not in input_shapes:
                init(name, arr)


        targetNetwork = softmax.simple_bind(ctx=self.ctx, data=data_shape)
        # ===============Initialization=================
        # First we get handle to input arrays
        arg_arrays = dict(zip(softmax.list_arguments(), targetNetwork.arg_arrays))

        # We initialize the weights with Xavier.
        init = mx.init.Xavier(factor_type="in")
        for name, arr in arg_arrays.items():
            if name not in input_shapes:
                init(name, arr)
        return network, targetNetwork

    def choose_action(self, state, epsilon):
        if file_logger.isEnabledFor(logging.DEBUG):
            action, value = self.greedy(state, verbose=True)
            file_logger.debug("model: " + str(self.model) + " current step value: " + str(value))
            if np.random.rand() < epsilon:
                return self.action_space.sample()
            else:
                return action
        else:
            if np.random.rand() < epsilon:
                return self.action_space.sample()
            else:
                return self.greedy(state)

    def greedy(self, state, verbose = False):
        # predict the Q values at current state
        state = state[np.newaxis,:]
        # replicate by batch_size
        state = np.tile(state, (self.sampleSize,1,1,1))

        q = self._network_forward(self.network, state)

        q = q[0,:]
        # keep track of current value
        if verbose :
            return np.argmax(q), np.max(q)
        else :
            # return the index of maximum Q value
            return np.argmax(q)

    def _network_forward(self, net, state):
        assert state.shape[0] == self.sampleSize
        assert state.shape[1] == self.input_shape[0]

        state_float = state / 255.0
        arg_arrays = net.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        net.forward(is_train=True)
        q = net.outputs[0].asnumpy()
        assert q.shape == (self.sampleSize, self.action_space_size)

        return q

    def train(self, state, action, reward, nextState, done):
        assert len(state.shape) == 4
        assert len(nextState.shape) == 4
        assert len(action.shape) == 1
        assert len(reward.shape) == 1
        assert len(done.shape) == 1
        assert state.shape == nextState.shape
        assert state.shape[0] == action.shape[0] == reward.shape[0] == nextState.shape[0] == done.shape[0]
        # ======================new version=========================================================================
        # if self.double_DQN :
        #     # selection step: use the online network to determine the greedy policy
        #     state_float = nextState / 255.0
        #     arg_arrays = self.network.arg_dict
        #     data = arg_arrays['data']
        #     data[:] = state_float
        #     self.network.forward(is_train=True)
        #     postq = self.network.outputs[0].asnumpy()
        #     postq = postq.transpose()
        #     assert postq.shape == (self.action_space_size, self.sampleSize)
        #
        #     # determine the greedy policy
        #     maxposta = np.argmax(postq, axis=0)
        #     assert maxposta.shape == (self.sampleSize,)
        #
        #     # evaluation step: use the target network to determine the future reward value
        #     arg_arrays = self.targetNetwork.arg_dict
        #     data = arg_arrays['data']
        #     data[:] = state_float
        #     self.targetNetwork.forward(is_train=True)
        #     postq = self.targetNetwork.outputs[0].asnumpy()
        #     postq = postq.transpose()
        #     assert postq.shape == (self.action_space_size, self.sampleSize)
        #     # use the online network's result to determine the future reward value
        #     maxpostq = postq[maxposta, range(self.sampleSize)]
        #     maxpostq = maxpostq[np.newaxis, :]
        #     assert maxpostq.shape == (1, self.sampleSize)
        # else :
        # feed-forward pass for poststates to get Q-values
        state_float = nextState / 255.0
        arg_arrays = self.targetNetwork.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        self.targetNetwork.forward(is_train=True)
        postq = self.targetNetwork.outputs[0].asnumpy()
        postq = postq.transpose()
        assert postq.shape == (self.action_space_size, self.sampleSize)

        # calculate max Q-value for each poststate
        maxpostq = np.max(postq, axis=0, keepdims=True)
        assert maxpostq.shape == (1, self.sampleSize)

        # ============================================================================
        # feed-forward pass for prestates
        state_float = state / 255.0
        arg_arrays = self.network.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        self.network.forward(is_train=True)
        q = self.network.outputs[0]
        preq = q.asnumpy().transpose()
        assert preq.shape == (self.action_space_size, self.sampleSize)

        # make copy of prestate Q-values as targets
        targets = preq.copy()
        old_targets = preq.copy()

        # clip rewards between -1 and 1
        rewards = np.clip(reward, -1, 1)

        # update Q-value targets for actions taken
        for i, action in enumerate(action):
            if done[i]:
                targets[action, i] = float(rewards[i])
            else:
                targets[action, i] = float(rewards[i]) + self.gamma * maxpostq[0, i]

        # calculate errors
        deltas = preq - targets
        assert deltas.shape == (self.action_space_size, self.sampleSize)

        # calculate cost, just in case
        cost = np.sum(deltas * deltas, keepdims=True)
        assert cost.shape == (1, 1)

        # clip errors
        if 1: # self.clip_error:
            # self.be.clip(deltas, -self.clip_error, self.clip_error, out = deltas)
            deltas = np.clip(deltas, -1, 1)

        # perform back-propagation of gradients
        label = arg_arrays['softmax_label']
        label[:] = (preq - deltas).transpose()
        self.network.backward()

        # perform optimization
        for i, pair in enumerate(zip(self.network.arg_arrays, self.network.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)
        # self.metric.update(label, self.network.outputs)


        if self.steps % 500 == 0 and logger.isEnabledFor(logging.DEBUG):
            np.set_printoptions(precision=4)
            logger.debug("old target: " + str(old_targets.transpose()))
            logger.debug("target: " + str(targets.transpose()))
            logger.debug("delta: " + str(deltas.transpose()))
            # logger.debug("sum(self.nextStates-self.mem.screens): " + str(np.sum(self.nextStates - self.mem.screens)))
            logger.debug("steps: " + str(self.steps))

        #sync target-network with network as mentioned in Mnih et al. Nature 2015
        if self.steps % self.targetNetC == 0:
            self.targetNetwork.copy_params_from(self.network.arg_dict, self.network.aux_dict)

    def ConvFactory(self, data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
        # there is an optional parameter ```wrokshpace``` may influece convolution performance
        # default, the workspace is set to 256(MB)
        # you may set larger value, but convolution layer only requires its needed but not exactly
        # MXNet will handle reuse of workspace without parallelism conflict
        conv = mx.symbol.Convolution(data=data, workspace=512,
                                     num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
        # bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = conv, act_type=act_type)
        return act

    def save_network(self, prefix='tmp_network', epoch=0):
        save_dict = {('network_arg:%s' % k) : v for k, v in self.network.arg_dict.items()}
        save_dict.update({('network_aux:%s' % k) : v for k, v in self.network.aux_dict.items()})
        save_dict.update({('targetNetwork_arg:%s' % k): v for k, v in self.targetNetwork.arg_dict.items()})
        save_dict.update({('targetNetwork_aux:%s' % k): v for k, v in self.targetNetwork.aux_dict.items()})
        param_name = '%s-%04d.params' % (prefix, epoch)
        mx.nd.save(param_name, save_dict)
        hyper_params = {}
        hyper_params['steps'] = self.steps
        with open('%s-%04d.json' % (prefix, epoch), 'w') as fp:
            json.dump(hyper_params, fp)
        logger.info('Saved checkpoint to \"%s\"', param_name)

    def load_network(self, prefix='tmp_network', epoch=0):
        save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'network_arg':
                arg_params[name] = v
            if tp == 'network_aux':
                aux_params[name] = v
        self.network.copy_params_from(arg_params, aux_params)

        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'targetNetwork_arg':
                arg_params[name] = v
            if tp == 'targetNetwork_aux':
                aux_params[name] = v
        self.targetNetwork.copy_params_from(arg_params, aux_params)
        with open('%s-%04d.json' % (prefix, epoch), 'r') as fp:
            hyper_params = json.load(fp)
        self.steps = hyper_params['steps']
        logger.info("resume from steps: " + str(self.steps))