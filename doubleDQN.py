from DQN import DQN
import numpy as np
import logging
logger = logging.getLogger(__name__)
file_logger = logging.getLogger('file')


class DoubleDQN(DQN):

    def __init__(self, args, action_space):
        super(DoubleDQN, self).__init__(args, action_space)

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
        state_float = nextState / 255.0
        arg_arrays = self.network.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        self.network.forward(is_train=True)
        postq = self.network.outputs[0].asnumpy()
        postq = postq.transpose()
        assert postq.shape == (self.action_space_size, self.sampleSize)

        # determine the greedy policy
        maxposta = np.argmax(postq, axis=0)
        assert maxposta.shape == (self.sampleSize,)

        # evaluation step: use the target network to determine the future reward value
        arg_arrays = self.targetNetwork.arg_dict
        data = arg_arrays['data']
        data[:] = state_float
        self.targetNetwork.forward(is_train=True)
        postq = self.targetNetwork.outputs[0].asnumpy()
        postq = postq.transpose()
        assert postq.shape == (self.action_space_size, self.sampleSize)
        # use the online network's result to determine the future reward value
        maxpostq = postq[maxposta, range(self.sampleSize)]
        maxpostq = maxpostq[np.newaxis, :]
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