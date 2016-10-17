from DQN import DQN
from doubleDQN import DoubleDQN
import mxnet as mx
import numpy as np
import logging
logger = logging.getLogger(__name__)
file_logger = logging.getLogger('file')


class DuelNetwork(DoubleDQN):

    def __init__(self, args, action_space):
        super(DuelNetwork, self).__init__(args, action_space)

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

        advantage_fc1 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="advantage_fc1")
        # bn1 = mx.sym.BatchNorm(data=fc1, name="bn1")
        advantage_act1 = mx.sym.Activation(data=advantage_fc1, name="advantage_act1", act_type="relu")
        advantage_fc2 = mx.sym.FullyConnected(data=advantage_act1, name="advantage_fc2", num_hidden=self.action_space_size)
        # softmax = mx.sym.Softmax(data=fc2, name="softmax")
        advantage_linear = mx.sym.LinearRegressionOutput(data=advantage_fc2, name="advantage_linear")

        value_fc1 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="value_fc1")
        value_act1 = mx.sym.Activation(data=value_fc1, name="value_act1", act_type="relu")
        value_fc2 = mx.sym.FullyConnected(data=value_act1, name="value_fc2", num_hidden=1)
        value_linear = mx.sym.LinearRegressionOutput(data=value_fc2, name="value_linear")

        linear = mx.sym.Group([value_linear, advantage_linear])

        # visualize the network
        batch_size = self.sampleSize
        data_shape = (batch_size, ) + self.input_shape
        # print data_shape
        logger.info( str(data_shape) )
        mx.viz.plot_network(linear, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'})
        # print softmax.list_arguments()
        logger.info( str(linear.list_arguments()) )
        # ==================Binding=====================
        # The symbol we created is only a graph description.
        # To run it, we first need to allocate memory and create an executor by 'binding' it.
        # In order to bind a symbol, we need at least two pieces of information: context and input shapes.
        # Context specifies which device the executor runs on, e.g. cpu, GPU0, GPU1, etc.
        # Input shapes define the executor's input array dimensions.
        # MXNet then run automatic shape inference to determine the dimensions of intermediate and output arrays.

        # We use simple_bind to let MXNet allocate memory for us.
        # You can also allocate memory youself and use bind to pass it to MXNet.
        network = linear.simple_bind(ctx=self.ctx, data=data_shape)

        #create a dumpy dataset
        # train_data = np.zeros((batch_size,) + self.input_shape)
        # label_data = np.zeros((batch_size, self.action_space_size))
        # train_iter = mx.io.NDArrayIter(data=train_data, label=label_data, batch_size=batch_size, shuffle=True)
        # input_shapes = dict(train_iter.provide_data + train_iter.provide_label)
        # print "input_shapes:", input_shapes
        input_shapes = dict([('data', data_shape)]  +
                            [('advantage_linear_label', (self.sampleSize, self.action_space_size))] +
                            [('value_linear_label', (self.sampleSize, 1))])
        logger.info("input_shapes:" + str(input_shapes))
        # ===============Initialization=================
        # First we get handle to input arrays
        arg_arrays = dict(zip(linear.list_arguments(), network.arg_arrays))

        # We initialize the weights with Xavier.
        init = mx.init.Xavier(factor_type="in")
        for name, arr in arg_arrays.items():
            if name not in input_shapes:
                init(name, arr)


        targetNetwork = linear.simple_bind(ctx=self.ctx, data=data_shape)
        # ===============Initialization=================
        # First we get handle to input arrays
        arg_arrays = dict(zip(linear.list_arguments(), targetNetwork.arg_arrays))

        # We initialize the weights with Xavier.
        init = mx.init.Xavier(factor_type="in")
        for name, arr in arg_arrays.items():
            if name not in input_shapes:
                init(name, arr)
        return network, targetNetwork

    def _network_forward(self, net, state, is_train=False):
        assert state.shape[0] == self.sampleSize
        assert state.shape[1] == self.input_shape[0]

        state_float = state / 255.0
        arg_arrays = net.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        net.forward(is_train=is_train)
        value = net.outputs[0].asnumpy()
        assert value.shape == (self.sampleSize, 1)
        advantage = net.outputs[1].asnumpy()
        assert advantage.shape == (self.sampleSize, self.action_space_size)

        q = value + advantage - np.mean(advantage, axis=1, keepdims=True)
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
        # selection step: use the online network to determine the greedy policy

        # feed-forward pass for poststates to get Q-values
        # state_float = nextState / 255.0
        # arg_arrays = self.network.arg_dict
        # data = arg_arrays['data']
        # data[:] = state_float
        # self.network.forward(is_train=True)
        # postq = self.network.outputs[0].asnumpy()
        postq = self._network_forward(self.network, nextState)
        postq = postq.transpose()
        assert postq.shape == (self.action_space_size, self.sampleSize)

        # determine the greedy policy
        maxposta = np.argmax(postq, axis=0)
        assert maxposta.shape == (self.sampleSize,)

        # evaluation step: use the target network to determine the future reward value

        # arg_arrays = self.targetNetwork.arg_dict
        # data = arg_arrays['data']
        # data[:] = state_float
        # self.targetNetwork.forward(is_train=True)
        # postq = self.targetNetwork.outputs[0].asnumpy()
        postq = self._network_forward(self.targetNetwork, nextState)
        postq = postq.transpose()
        assert postq.shape == (self.action_space_size, self.sampleSize)
        # use the online network's result to determine the future reward value
        maxpostq = postq[maxposta, range(self.sampleSize)]
        maxpostq = maxpostq[np.newaxis, :]
        assert maxpostq.shape == (1, self.sampleSize)
        # ============================================================================
        # feed-forward pass for prestates
        # state_float = state / 255.0
        # arg_arrays = self.network.arg_dict
        # data = arg_arrays['data']
        # data[:] = state_float
        # self.network.forward(is_train=True)
        # q = self.network.outputs[0]
        # preq = q.asnumpy().transpose()
        q = self._network_forward(self.network, state, is_train=True)
        preq = q.transpose()
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
        arg_arrays = self.network.arg_dict
        value_label = arg_arrays['value_linear_label']
        advantage_label = arg_arrays['advantage_linear_label']
        targets_clipped = preq - deltas

        # target value
        value = np.mean(targets_clipped, axis=0)
        value_label[:] = value

        # target advantage
        advantage_label[:] = (targets_clipped - value).transpose()
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