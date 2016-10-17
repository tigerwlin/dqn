import mxnet as mx
import numpy as np
import logging
import pprint

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    # there is an optional parameter ```wrokshpace``` may influece convolution performance
    # default, the workspace is set to 256(MB)
    # you may set larger value, but convolution layer only requires its needed but not exactly
    # MXNet will handle reuse of workspace without parallelism conflict
    conv = mx.symbol.Convolution(data=data, workspace=256,
                                 num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act
# we can use mx.sym in short of mx.symbol
data = mx.sym.Variable("data")

conv1 = ConvFactory(data=data, kernel=(8,8), stride=(4,4), pad=(1,1), num_filter=32, act_type="relu")
conv2 = ConvFactory(data=conv1, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64, act_type="relu")
conv3 = ConvFactory(data=conv2, kernel=(3,3), stride=(1,1), pad=(1,1), num_filter=64, act_type="relu")

fc1 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="fc1")
bn1 = mx.sym.BatchNorm(data=fc1, name="bn1")
act1 = mx.sym.Activation(data=bn1, name="act1", act_type="relu")
fc2 = mx.sym.FullyConnected(data=act1, name="fc2", num_hidden=10)
# softmax = mx.sym.Softmax(data=fc2, name="softmax")
# linear = mx.sym.LinearRegressionOutput(data=fc2, name="softmax")
linear1 = mx.sym.LinearRegressionOutput(data=fc2, name="linear1")

fc1_2 = mx.sym.FullyConnected(data=conv3, num_hidden=512, name="fc1_2")
bn1_2 = mx.sym.BatchNorm(data=fc1_2, name="bn1_2")
act1_2 = mx.sym.Activation(data=bn1_2, name="act1_2", act_type="relu")
fc2_2 = mx.sym.FullyConnected(data=act1_2, name="fc2_2", num_hidden=10)
linear2 = mx.sym.LinearRegressionOutput(data=fc2_2, name="linear2")

linear = mx.sym.Group([linear1, linear2])
# visualize the network
batch_size = 100
data_shape = (batch_size, 1, 28, 28)
# mx.viz.plot_network(softmax, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'})
# print 'softmax: ', softmax.list_arguments()
mx.viz.plot_network(linear, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'})
print 'linear: ', linear.list_arguments()

#=======================================================================================


def SGD(key, weight, grad, lr=0.1, grad_norm=batch_size):
    # key is key for weight, we can customize update rule
    # weight is weight array
    # grad is grad array
    # lr is learning rate
    # grad_norm is scalar to norm gradient, usually it is batch_size
    norm = 1.0 / grad_norm
    # here we can bias' learning rate 2 times larger than weight
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
    else:
        pass

# We use utils function in sklearn to get MNIST dataset in pickle
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
mnist = fetch_mldata('MNIST original', data_home="./data")
# shuffle data
X, y = shuffle(mnist.data, mnist.target)
Y = np.zeros((y.shape[0], 10))
for i in xrange(y.shape[0]):
    Y[i,y[i]] = 1
y = Y
X = np.reshape(X, (-1,28,28))
X = X[:,np.newaxis,:,:]
# split dataset
train_data = X[:50000, :].astype('float32')
train_label = y[:50000]
val_data = X[50000: 60000, :].astype('float32')
val_label = y[50000:60000]
# Normalize data
train_data[:] /= 256.0
val_data[:] /= 256.0
# Build iterator
train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=batch_size)

# def Accuracy(label, pred_prob):
def Accuracy(target, pred_prob):
    label = np.argmax(target, axis=1)
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(label == pred) * 1.0 / label.shape[0]

#=======================================================================================
# # We will make model with current current symbol
# # For demo purpose, this model only train 1 epoch
# # We will use the first GPU to do training
# num_epoch = 10
# model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
#                              learning_rate=0.05, momentum=0.9, wd=0.00001)
# model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch, optimizer='RMSProp',
#                              learning_rate=0.002, wd=0.00001)
#
# model.fit(X=train_iter,
#           eval_data=val_iter,
#           eval_metric="accuracy",
#           batch_end_callback=mx.callback.Speedometer(batch_size))

# ==================Binding=====================
# # context different to ```mx.model```,
# # In mx.model, we wrapped parameter server, but for a single executor, the context is only able to be ONE device
# # run on cpu
# # ctx = mx.cpu()
# # run on gpu
# ctx = mx.gpu()
# # run on third gpu
# # ctx = mx.gpu(2)
# executor = softmax.simple_bind(ctx=ctx, data=data_shape) #, grad_req='write')
# # The default ctx is CPU, data's shape is required and ```simple_bind``` will try to infer all other required
# # For MLP, the ```grad_req``` is write to, and for RNN it is different
#
#
#
#
# # args = executor.arg_dict
# # Equivalently you could do this:
# args = dict(zip(softmax.list_arguments(), executor.arg_arrays))
# grads = executor.grad_dict
# aux_states = executor.aux_dict
#
# # For outputs we need to assemble the dict by hand:
# outputs = dict(zip(softmax.list_outputs(), executor.outputs))
#
# # we can print the args we have
# print("args: %s" % pprint.pformat(args))
# print("-" * 20)
# print("grads: %s" % pprint.pformat(grads))
# print("-" * 20)
# print("aux_states: %s" % pprint.pformat(aux_states))
# print("-" * 20)
# print("outputs: %s" % pprint.pformat(outputs))
#
# args['fc1_weight'][:] = mx.random.uniform(-0.07, 0.07, args['fc1_weight'].shape)
# args['fc2_weight'][:] = np.random.uniform(-0.07, 0.07, args['fc2_weight'].shape)  # equivalent
# args['bn1_beta'][:] = 1.0
# args['bn1_gamma'][:] = 1.0
# args['fc1_bias'][:] = 0
# args['fc2_bias'][:] = 0
# # Don't initialize data or softmax_label

# ==================Binding=====================
# The symbol we created is only a graph description.
# To run it, we first need to allocate memory and create an executor by 'binding' it.
# In order to bind a symbol, we need at least two pieces of information: context and input shapes.
# Context specifies which device the executor runs on, e.g. cpu, GPU0, GPU1, etc.
# Input shapes define the executor's input array dimensions.
# MXNet then run automatic shape inference to determine the dimensions of intermediate and output arrays.

# data iterators defines shapes of its output with provide_data and provide_label property.
# input_shapes = dict(train_iter.provide_data + train_iter.provide_label)
input_shapes = dict(train_iter.provide_data + [('linear1_label', (100, 10))] + [('linear2_label', (100, 10))])
print 'input_shapes', input_shapes
# We use simple_bind to let MXNet allocate memory for us.
# You can also allocate memory youself and use bind to pass it to MXNet.
# exe = softmax.simple_bind(ctx=mx.gpu(0), **input_shapes)
# exe2 = softmax.simple_bind(ctx=mx.gpu(0), **input_shapes)
# exe = linear.simple_bind(ctx=mx.gpu(0), **input_shapes)
exe = linear.simple_bind(ctx=mx.gpu(0), data=data_shape)
# ===============Initialization=================
# First we get handle to input arrays
arg_arrays = exe.arg_dict
data = arg_arrays[train_iter.provide_data[0][0]]
# label = arg_arrays[train_iter.provide_label[0][0]]
label1 = arg_arrays['linear1_label']
label2 = arg_arrays['linear2_label']
# We initialize the weights with uniform distribution on (-0.01, 0.01).
init = mx.init.Uniform(scale=0.01)
for name, arr in arg_arrays.items():
    if name not in input_shapes:
        init(name, arr)

# We also need to create an optimizer for updating weights
# opt = mx.optimizer.SGD(
#     learning_rate=0.1,
#     momentum=0.9,
#     wd=0.00001,
#     rescale_grad=1.0 / train_iter.batch_size)
opt = mx.optimizer.RMSProp(
    learning_rate=0.002,
    wd=0.00001)

updater = mx.optimizer.get_updater(opt)

# Finally we need a metric to print out training progress
metric = mx.metric.Accuracy()

# Training loop begines
for epoch in range(2):
    train_iter.reset()
    val_iter.reset()
    metric.reset()
    t = 0
    for batch in train_iter:
        # Copy data to executor input. Note the [:].
        data[:] = batch.data[0]
        label1[:] = batch.label[0]
        label2[:] = batch.label[0]

        # Forward
        exe.forward(is_train=True)

        # You perform operations on exe.outputs here if you need to.
        # For example, you can stack a CRF on top of a neural network.

        # Backward
        exe.backward()

        # Update
        for i, pair in enumerate(zip(exe.arg_arrays, exe.grad_arrays)):
            weight, grad = pair
            updater(i, grad, weight)
        # metric.update(batch.label, exe.outputs)
        t += 1
        if t % 100 == 0:
            print 'epoch:', epoch, 'train iter:', t, 'metric:', Accuracy(label1.asnumpy(), exe.outputs[0].asnumpy())
            print 'epoch:', epoch, 'train iter:', t, 'metric:', Accuracy(label2.asnumpy(), exe.outputs[1].asnumpy())
        #     print 'epoch:', epoch, 'train iter:', t, 'metric:', metric.get()
    t = 0
    # metric.reset()
    for batch in val_iter:
        # Copy data to executor input. Note the [:].
        data[:] = batch.data[0]
        label1[:] = batch.label[0]
        label2[:] = batch.label[0]
        # Forward
        exe.forward(is_train=False)
        # metric.update(batch.label, exe.outputs)
        t += 1
        if t % 50 == 0:
            print 'epoch:', epoch, 'test iter:', t, 'metric:', Accuracy(label1.asnumpy(), exe.outputs[0].asnumpy())
            print 'epoch:', epoch, 'test iter:', t, 'metric:', Accuracy(label2.asnumpy(), exe.outputs[1].asnumpy())
        #     print 'epoch:', epoch, 'test iter:', t, 'metric:', metric.get()
#=========================================================================
# for batch in val_iter:
#     # Copy data to executor input. Note the [:].
#     data[:] = batch.data[0]
#     label[:] = batch.label[0]
#
#     # Forward
#     exe2.forward(is_train=False)
#     # metric.update(batch.label, exe2.outputs)
#     t += 1
#     if t % 50 == 0:
#         print 'epoch:', epoch, 'test iter:', t, 'metric:', Accuracy(label.asnumpy(), exe2.outputs[0].asnumpy())
#     #     print 'epoch:', epoch, 'test iter:', t, 'metric:', metric.get()

#=========================================================================
# num_round = 2
# keys = softmax.list_arguments()
# # we use extra ndarray to save output of net
# pred_prob = mx.nd.zeros(executor.outputs[0].shape)
# for roundNo in range(num_round):
#     train_iter.reset()
#     val_iter.reset()
#     train_acc = 0.
#     val_acc = 0.
#     nbatch = 0.
#     # train
#     for dbatch in train_iter:
#         data = dbatch.data[0]
#         label = dbatch.label[0]
#         # copy data into args
#         args["data"][:] = data # or we can ```data.copyto(args["data"])```
#         args["softmax_label"][:] = label
#         executor.forward(is_train=True)
#         pred_prob[:] = executor.outputs[0]
#         executor.backward()
#         for key in keys:
#             SGD(key, args[key], grads[key])
#         # Update
#         # for i, pair in enumerate(zip(executor.arg_arrays, executor.grad_arrays)):
#         #     weight, grad = pair
#         #     updater(i, grad, weight)
#         train_acc += Accuracy(label.asnumpy(), pred_prob.asnumpy())
#         nbatch += 1.
#     logging.info("Finish training iteration %d" % roundNo)
#     train_acc /= nbatch
#     nbatch = 0.
#     # eval
#     for dbatch in val_iter:
#         data = dbatch.data[0]
#         label = dbatch.label[0]
#         args["data"][:] = data
#         executor.forward(is_train=False)
#         pred_prob[:] = executor.outputs[0]
#         val_acc += Accuracy(label.asnumpy(), pred_prob.asnumpy())
#         nbatch += 1.
#     val_acc /= nbatch
#     logging.info("Train Acc: %.4f" % train_acc)
#     logging.info("Val Acc: %.4f" % val_acc)
#=========================================================================

