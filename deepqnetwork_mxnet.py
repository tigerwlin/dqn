from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Xavier
from neon.initializers import Uniform
from neon.optimizers import RMSProp, Adam, Adadelta
from neon.layers import Affine, Conv, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
from neon.util.persist import save_obj
import numpy as np
import mxnet as mx
import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, num_actions, args):
    # remember parameters
    self.num_actions = num_actions
    self.batch_size = args.batch_size
    self.discount_rate = args.discount_rate
    self.history_length = args.history_length
    self.screen_dim = (args.screen_height, args.screen_width)
    self.clip_error = args.clip_error
    self.min_reward = args.min_reward
    self.max_reward = args.max_reward
    self.batch_norm = args.batch_norm

    # create Neon backend
    # self.be = gen_backend(backend = args.backend,
    #              batch_size = args.batch_size,
    #              rng_seed = args.random_seed,
    #              device_id = args.device_id,
    #              datatype = np.dtype(args.datatype).type,
    #              stochastic_round = args.stochastic_round)

    # prepare tensors once and reuse them
    self.input_shape = (self.history_length,) + self.screen_dim # + (self.batch_size,)
    # self.input = self.be.empty(self.input_shape)
    # self.input.lshape = self.input_shape # HACK: needed for convolutional networks
    # self.targets = self.be.empty((self.num_actions, self.batch_size))

    # create model
    self.network, self.targetNetwork = self.build_network()
    # layers = self._createLayers(num_actions)
    # self.model = Model(layers = layers)
    # self.cost = GeneralizedCost(costfunc = SumSquared())
    # Bug fix
    # for l in self.model.layers.layers:
    #   l.parallelism = 'Disabled'
    # self.model.initialize(self.input_shape[:-1], self.cost)
    # if args.optimizer == 'rmsprop':
    #   self.optimizer = RMSProp(learning_rate = args.learning_rate,
    #       decay_rate = args.decay_rate,
    #       stochastic_round = args.stochastic_round)
    # elif args.optimizer == 'adam':
    #   self.optimizer = Adam(learning_rate = args.learning_rate,
    #       stochastic_round = args.stochastic_round)
    # elif args.optimizer == 'adadelta':
    #   self.optimizer = Adadelta(decay = args.decay_rate,
    #       stochastic_round = args.stochastic_round)
    # else:
    #   assert false, "Unknown optimizer"

    self.opt = mx.optimizer.RMSProp(
            learning_rate= args.learning_rate, rescale_grad = 1.0/self.batch_size,
            gamma2 = 0)
    self.updater = mx.optimizer.get_updater(self.opt)
    self.metric = mx.metric.MSE()
    self.metric.reset()

    # create target model
    self.train_iterations = 0
    # if args.target_steps:
    #   self.target_model = Model(layers = self._createLayers(num_actions))
    #   # Bug fix
    #   for l in self.target_model.layers.layers:
    #     l.parallelism = 'Disabled'
    #   self.target_model.initialize(self.input_shape[:-1])
    #   self.save_weights_prefix = args.save_weights_prefix
    # else:
    #   self.target_model = self.model

    self.callback = None

  def _createLayers(self, num_actions):
    # create network
    init_xavier_conv = Xavier(local=True)
    init_xavier_affine = Xavier(local=False)
    # init_uniform_conv = Uniform(low=-.01, high=.01)
    # init_uniform_affine = Uniform(low=-.01, high=.01)
    layers = []
    # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
    # layers.append(Conv((8, 8, 32), strides=4, init=init_xavier_conv, activation=Rectlin(), batch_norm=self.batch_norm))
    layers.append(Conv((5, 5, 32), strides=2, init=init_xavier_conv, activation=Rectlin(), batch_norm=self.batch_norm))
    # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
    # layers.append(Conv((4, 4, 64), strides=2, init=init_xavier_conv, activation=Rectlin(), batch_norm=self.batch_norm))
    layers.append(Conv((5, 5, 32), strides=2, init=init_xavier_conv, activation=Rectlin(), batch_norm=self.batch_norm))
    # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
    # layers.append(Conv((3, 3, 64), strides=1, init=init_xavier_conv, activation=Rectlin(), batch_norm=self.batch_norm))
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    layers.append(Affine(nout=512, init=init_xavier_affine, activation=Rectlin(), batch_norm=self.batch_norm))
    # The output layer is a fully-connected linear layer with a single output for each valid action.
    layers.append(Affine(nout=num_actions, init = init_xavier_affine))
    return layers

  def _setInput(self, states):
    # change order of axes to match what Neon expects
    states = np.transpose(states, axes = (1, 2, 3, 0))
    # copy() shouldn't be necessary here, but Neon doesn't work otherwise
    self.input.set(states.copy())
    # normalize network input between 0 and 1
    self.be.divide(self.input, 255, self.input)

  def update_target_network(self):
      # have to serialize also states for batch normalization to work
      # pdict = self.model.get_description(get_weights=True, keep_states=True)
      # self.target_model.deserialize(pdict, load_states=True)
      self.targetNetwork.copy_params_from(self.network.arg_dict, self.network.aux_dict)

  def train(self, minibatch, epoch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    # feed-forward pass for poststates to get Q-values
    # self._setInput(poststates)
    # postq = self.target_model.fprop(self.input, inference = True)
    state_float = poststates / 255.0
    arg_arrays = self.targetNetwork.arg_dict
    data = arg_arrays['data']
    data[:] = state_float
    self.targetNetwork.forward(is_train=True)
    postq = self.targetNetwork.outputs[0].asnumpy()
    postq = postq.transpose()
    assert postq.shape == (self.num_actions, self.batch_size)

    # calculate max Q-value for each poststate
    # maxpostq = self.be.max(postq, axis=0).asnumpyarray()
    maxpostq = np.max(postq, axis=0, keepdims=True)
    assert maxpostq.shape == (1, self.batch_size)

    # feed-forward pass for prestates
    # self._setInput(prestates)
    # preq = self.model.fprop(self.input, inference = False)
    state_float = prestates / 255.0
    arg_arrays = self.network.arg_dict
    data = arg_arrays['data']
    data[:] = state_float
    self.network.forward(is_train=True)
    q = self.network.outputs[0]
    preq = q.asnumpy().transpose()
    assert preq.shape == (self.num_actions, self.batch_size)

    # make copy of prestate Q-values as targets
    # targets = preq.asnumpyarray().copy()
    targets = preq.copy()

    # old_targets = preq.asnumpyarray().copy()
    old_targets = preq.copy()

    # =================================================================
    # replace this with my precomputed reward


    # =================================================================
    # clip rewards between -1 and 1
    rewards = np.clip(rewards, self.min_reward, self.max_reward)

    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        targets[action, i] = float(rewards[i])
      else:
        targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]

    # =================================================================
    # rewards = np.clip(rewards, -1, 1)
    #
    # future_Qvalue = maxpostq[0]
    # future_reward = future_Qvalue #np.max(future_Qvalue, axis=1)
    # # future_reward = future_reward[:, np.newaxis]
    #
    # rewards += (1-terminals)*self.discount_rate*future_reward
    # # reward += (1 - abs(reward)) * self.gamma * future_reward
    #
    # target_reward =targets
    # # old_target_reward = copy.deepcopy(target_reward)
    # for i in xrange(self.batch_size):
    #   # target_reward[i][action[i]] = reward[i]
    #   # clip error to [-1, 1], Mnih 2015 Nature
    #   target_reward[actions[i]][i] = max(min(rewards[i], target_reward[actions[i]][i] + 1),
    #                                     target_reward[actions[i]][i] - 1)
    # targets = target_reward
    # =================================================================

    # copy targets to GPU memory
    # self.targets.set(targets)

    # calculate errors
    # deltas = self.cost.get_errors(preq, self.targets)
    deltas = preq - targets
    assert deltas.shape == (self.num_actions, self.batch_size)
    #assert np.count_nonzero(deltas.asnumpyarray()) == 32

    # calculate cost, just in case
    # cost = self.cost.get_cost(preq, self.targets)
    cost = np.sum(deltas*deltas, keepdims=True)
    assert cost.shape == (1,1)

    # clip errors
    if self.clip_error:
      # self.be.clip(deltas, -self.clip_error, self.clip_error, out = deltas)
      deltas = np.clip(deltas, -self.clip_error, self.clip_error)

    # perform back-propagation of gradients
    # self.model.bprop(deltas)
    label = arg_arrays['softmax_label']
    label[:] = (preq - deltas).transpose()
    self.network.backward()

    # perform optimization
    # self.optimizer.optimize(self.model.layers_to_optimize, epoch)

    for i, pair in enumerate(zip(self.network.arg_arrays, self.network.grad_arrays)):
      weight, grad = pair
      self.updater(i, grad, weight)
    # self.metric.update(label, self.network.outputs)

    # increase number of weight updates (needed for stats callback)
    self.train_iterations += 1

    if self.train_iterations % 500 == 0:
      np.set_printoptions(precision=4)
      print "old target: ", old_targets.transpose()
      print "target: ", targets.transpose()
      print "delta: ", deltas.transpose()
      # print "delta2: ", (targets - old_targets).transpose()
      print "iter: ", self.train_iterations

      # calculate statistics
    if self.callback:
      self.callback.on_train(cost[0,0])

    return old_targets, targets


  def predict(self, states):
    # minibatch is full size, because Neon doesn't let change the minibatch size
    assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

    # calculate Q-values for the states
    # self._setInput(states)
    # qvalues = self.model.fprop(self.input, inference = True)
    state_float = states / 255.0
    arg_arrays = self.network.arg_dict
    data = arg_arrays['data']
    data[:] = state_float
    self.network.forward(is_train=True)
    q = self.network.outputs[0]
    qvalues = q.asnumpy().transpose()
    assert qvalues.shape == (self.num_actions, self.batch_size)
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Q-values: " + str(qvalues[:,0]))

    # transpose the result, so that batch size is first dimension
    return qvalues.transpose()

  def load_weights(self, load_path):
    self.model.load_params(load_path)

  def save_weights(self, save_path):
    self.model.save_params(save_path)


# ====================mxnet====================================================
  def build_network(self):
    # we can use mx.sym in short of mx.symbol
    data = mx.sym.Variable("data")

    # conv1 = self.ConvFactory(data=data, kernel=(8,8), stride=(4,4), pad=(1,1), num_filter=32, act_type="relu")
    # conv2 = self.ConvFactory(data=conv1, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64, act_type="relu")
    # conv3 = self.ConvFactory(data=conv2, kernel=(3,3), stride=(1,1), pad=(1,1), num_filter=64, act_type="relu")
    # ====================================================================================================
    conv1 = self.ConvFactory(data=data, kernel=(5, 5), stride=(2, 2), pad=(0, 0), num_filter=32, act_type="relu")
    # conv2 = self.ConvFactory(data=conv1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64, act_type="relu")
    conv3 = self.ConvFactory(data=conv1, kernel=(5, 5), stride=(2, 2), pad=(0, 0), num_filter=32, act_type="relu")

    fc1 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="fc1")
    # bn1 = mx.sym.BatchNorm(data=fc1, name="bn1")
    act1 = mx.sym.Activation(data=fc1, name="act1", act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name="fc2", num_hidden=self.num_actions)
    # softmax = mx.sym.Softmax(data=fc2, name="softmax")
    softmax = mx.sym.LinearRegressionOutput(data=fc2, name="softmax")

    # visualize the network
    batch_size = self.batch_size
    data_shape = (batch_size,) + self.input_shape
    print data_shape
    mx.viz.plot_network(softmax, shape={"data": data_shape}, node_attrs={"shape": 'oval', "fixedsize": 'false'})
    print softmax.list_arguments()

    # ==================Binding=====================
    # The symbol we created is only a graph description.
    # To run it, we first need to allocate memory and create an executor by 'binding' it.
    # In order to bind a symbol, we need at least two pieces of information: context and input shapes.
    # Context specifies which device the executor runs on, e.g. cpu, GPU0, GPU1, etc.
    # Input shapes define the executor's input array dimensions.
    # MXNet then run automatic shape inference to determine the dimensions of intermediate and output arrays.

    # We use simple_bind to let MXNet allocate memory for us.
    # You can also allocate memory youself and use bind to pass it to MXNet.
    network = softmax.simple_bind(ctx=mx.gpu(0), data=data_shape)
    # network = softmax.simple_bind(ctx=mx.cpu(), data=data_shape)

    # create a dumpy dataset
    train_data = np.zeros((batch_size,) + self.input_shape)
    label_data = np.zeros((batch_size, self.num_actions))
    train_iter = mx.io.NDArrayIter(data=train_data, label=label_data, batch_size=batch_size, shuffle=True)
    input_shapes = dict(train_iter.provide_data + train_iter.provide_label)
    print "input_shapes:", input_shapes
    # ===============Initialization=================
    # First we get handle to input arrays
    arg_arrays = dict(zip(softmax.list_arguments(), network.arg_arrays))

    # We initialize the weights with uniform distribution on (-0.01, 0.01).
    # init = mx.init.Uniform(scale=.01)
    init = mx.init.Xavier(factor_type="in")
    for name, arr in arg_arrays.items():
      if name not in input_shapes:
        init(name, arr)

    targetNetwork = softmax.simple_bind(ctx=mx.gpu(0), data=data_shape)
    # ===============Initialization=================
    # First we get handle to input arrays
    arg_arrays = dict(zip(softmax.list_arguments(), targetNetwork.arg_arrays))

    # We initialize the weights with uniform distribution on (-0.01, 0.01).
    init = mx.init.Uniform(scale=0.01)
    for name, arr in arg_arrays.items():
      if name not in input_shapes:
        init(name, arr)
    return network, targetNetwork

  # Basic Conv + BN + ReLU factory
  def ConvFactory(self, data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu"):
    # there is an optional parameter ```wrokshpace``` may influece convolution performance
    # default, the workspace is set to 256(MB)
    # you may set larger value, but convolution layer only requires its needed but not exactly
    # MXNet will handle reuse of workspace without parallelism conflict
    conv = mx.symbol.Convolution(data=data, workspace=256,
                                 num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    # bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data=conv, act_type=act_type)
    return act