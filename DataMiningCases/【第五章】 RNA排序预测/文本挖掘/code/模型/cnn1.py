#!/usr/bin/env python

import sys
import os
import time


import theano
import theano.tensor as T

import numpy as np
import lasagne
import cPickle
import lasagne.layers.dnn
#%%
# ##################### Build the neural network model #######################

def build_cnn(input1,input2):
    input1 = lasagne.layers.InputLayer(shape=(None, 1, 4, 174),
                                        input_var=input1)
    input2 = lasagne.layers.InputLayer(shape=(None, 3, 5, 174),
                                        input_var=input2)
    ###################### Layer One: ######################
    cnn1 = lasagne.layers.dnn.Conv2DDNNLayer(
            input1, num_filters=16, filter_size=(4, 21),
            stride=(1,1), nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    netshape = lasagne.layers.get_output_shape(cnn1)
    print('First structure Layer 1 Conv shape:')
    print(netshape)
    
    # Local Response Normalization:
    cnn1 = lasagne.layers.normalization.LocalResponseNormalization2DLayer(
            cnn1, alpha=0.001, k=1 , beta=0.75, n=21)
    inputlayer1 = lasagne.layers.dnn.MaxPool2DDNNLayer(
            cnn1, pool_size=(1, 4), stride=None)
    
    #jin
    netshape = lasagne.layers.get_output_shape(inputlayer1)
    print('First structure Layer 1 MaxPool shape:')
    print(netshape)
    
    dnn1 = lasagne.layers.DenseLayer(inputlayer1,num_units=10,nonlinearity=lasagne.nonlinearities.rectify)
    
    
    ###################### Layer One: ######################
    cnn2 = lasagne.layers.dnn.Conv2DDNNLayer(
            input2, num_filters=16, filter_size=(5, 21),
            stride=(1,1), nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    netshape = lasagne.layers.get_output_shape(cnn2)
    print('Secondary structure Layer 1 Conv shape:')
    print(netshape)
    
    # Local Response Normalization:
    cnn2 = lasagne.layers.normalization.LocalResponseNormalization2DLayer(
            cnn2, alpha=0.001, k=1 , beta=0.75, n=21)
    inputlayer2 = lasagne.layers.dnn.MaxPool2DDNNLayer(
            cnn2, pool_size=(1, 4), stride=None)
    
    #jin
    netshape = lasagne.layers.get_output_shape(inputlayer2)
    print('Secondary structure Layer 1 MaxPool shape:')
    print(netshape)
    
    dnn2 = lasagne.layers.DenseLayer(inputlayer2,num_units=10,nonlinearity=lasagne.nonlinearities.rectify)
    
    # merge two representation
    common = lasagne.layers.ConcatLayer([dnn1,dnn2])
    #jin
    netshape = lasagne.layers.get_output_shape(common)
    print('two representation shape:')
    print(netshape)
    common_dnn1 = lasagne.layers.DenseLayer(common,num_units=10,nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.DenseLayer(
            common_dnn1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    #jin
    netshape = lasagne.layers.get_output_shape(network)
    print('softmax shape:')
    print(netshape)

    return network


# ############################# Batch iterator ###############################
# Load data set
def loadArray(filename,batch_size):
    with file(filename,'rb') as f:
        first = cPickle.load(f)
        second = cPickle.load(f)
        label = cPickle.load(f)
    
    length = len(label)
    for start_id in range(0,length-batch_size+1,batch_size):
        if start_id + batch_size < length:
            yield first[start_id:start_id+batch_size,0:1,:,:],second[start_id:start_id+batch_size,:,:,:],label[start_id:start_id+batch_size]
        else:
            yield first[start_id:-1,0:1,:,:],second[start_id:-1,:,:,:],label[start_id:-1]

#%%
def weighted_cross_entropy(coding_dist, true_dist):
    return -T.sum(true_dist * T.log(coding_dist),axis=coding_dist.ndim - 1)

def one_hot_encode(label,classnum,weight):
    code = np.zeros((len(label),classnum),np.int32)
    for i in range(classnum):
        index = label==i
        code[index,i] = weight[i]
    return code
"""
a = T.dmatrix()
b = T.imatrix()
c = weighted_cross_entropy(a,b)
cost=theano.function([a,b],c)
aa = np.array([[.1,.9],[.2,.8],[.9,.1]],dtype=np.float32)
bb = np.array([1,1,0],np.int32)
onehot = one_hot_encode(bb,2,[1,2])
print onehot
cc = cost(aa,onehot)
print cc
"""

#%%
def detail_result(predict,label,classnum):
    assert len(predict)==len(label)
    results = np.zeros((classnum),np.int32)
    lengths = np.zeros((classnum),np.int32)
    for i in range(classnum):
        indices = label==i
        sublabel = label[indices]
        subpredict = predict[indices]
        results[i] = sum(sublabel==subpredict)
        lengths[i] = len(sublabel)
    return results,lengths
"""
predict=np.array([1,0,1,0,1,0,0,0,0,1])
label=np.array([1,0,1,0,1,0,1,0,1,0])
print detail_result(predict,label,2)
"""
#%%
# ############################## Main program ################################

def main(val_data,weight,paramFile="",num_epochs=100):
    # Prepare Theano variables for inputs and targets
    input1 = T.tensor4('input1')
    input2 = T.tensor4('input2')
    target_var = T.imatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input1,input2)
    
    #jin
    if paramFile=="":
		print("Train a new network!")
    else:
		print("Load well trained parameters from "+paramFile)
		f = file(paramFile,'rb')
		params = cPickle.load(f)
		f.close()
		lasagne.layers.set_all_param_values(network,params)
	
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = weighted_cross_entropy(prediction, target_var)
    loss = loss.mean()
    train_decode = T.argmax(prediction, axis=1)
    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # adjust learning rate here
    lr=theano.shared(np.array(0.1,dtype=theano.config.floatX))
    # if next step loss is larger, update lr as lr*lr_decay
    lr_decay=np.array(0.5,dtype=theano.config.floatX)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=lr, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = weighted_cross_entropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    test_decode = T.argmax(test_prediction, axis=1)
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input1, input2, target_var], [loss,train_decode], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input1, input2, target_var], [test_loss, test_decode])

    # Finally, launch the training loop.
    print("Starting training...")
    
    
    # train set and validation set
    dirpath = os.getcwd()
    print('dirpath = '+dirpath)
    
    batch_size = 500
    fileprefix = dirpath + '/data'
    prev_test_loss=100
    
    trainfile = 'cnn2inputTrain'
    testfile = 'cnn2inputTest'

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_results = np.zeros((2),np.int32)
        train_numbers = np.zeros((2),np.int32)
        train_loss = 0
        start_time = time.time()
        
        
        for batch in loadArray(trainfile,batch_size):
            first, second, targets = batch
            target_code = one_hot_encode(targets,2,weight)
            temp_loss,train_predict = train_fn(first, second, target_code)
            temp_result,temp_number = detail_result(train_predict,targets,2)
            #print("  batch train loss=",temp_loss)
            #print("  batch TP=%.4f"%(float(temp_result[1])/temp_number[1]))
            #print("  batch TN=%.4f"%(float(temp_result[0])/temp_number[0]))
            train_loss += temp_loss
            train_results += temp_result
            train_numbers += temp_number
                    
        print "epoch",epoch+1,"took time :%.3f"%(time.time() - start_time),"s"
        print "  Train Loss Per Sample =",float(train_loss)/sum(train_numbers)
        print("  Train TP=%.4f"%(float(train_results[1])/train_numbers[1]))
        print("  Train TN=%.4f"%(float(train_results[0])/train_numbers[0]))
        print("learning rate=",lr.get_value())

        # Then we print the results for this epoch:
        
        
        """
        # store parameters
        print("  should store epoch {}".format(epoch+1))
        pythonName,suffix = os.path.splitext(__file__)
        param2store = lasagne.layers.get_all_param_values(network)
        storename = pythonName+"_"+str((epoch+1))+"_accu="+str(train_err / train_batches * 100)+".save"
        with file(storename,'wb') as f:
            cPickle.dump(param2store,f)
            """
	
        # After training, we compute and print the test error:
        train_results = np.zeros((2),np.int32)
        train_numbers = np.zeros((2),np.int32)
        train_loss = 0
        
        #testfile=fileprefix+str(val_data)
        print("Test on"+testfile)
        #for batch in loadArray(train_dirpath):
        for batch in loadArray(testfile,batch_size):
            first, second, targets = batch
            target_code = one_hot_encode(targets,2,weight)
            temp_loss,train_predict = val_fn(first, second, target_code)
            temp_result,temp_number = detail_result(train_predict,targets,2)
            #print("  batch Test loss=",temp_loss)
            #print("  batch TP=%.4f"%(float(temp_result[1])/temp_number[1]))
            #print("  batch TN=%.4f"%(float(temp_result[0])/temp_number[0]))
            train_loss += temp_loss
            train_results += temp_result
            train_numbers += temp_number
        if prev_test_loss < float(train_loss)/sum(train_numbers):
            lr.set_value(lr.get_value()*lr_decay)
        prev_test_loss = float(train_loss)/sum(train_numbers)
        print "  Test Loss Per Sample =",float(train_loss)/sum(train_numbers)
        print("  Test TP=%.4f"%(float(train_results[1])/train_numbers[1]))
        print("  Test TN=%.4f"%(float(train_results[0])/train_numbers[0]))
 
 
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on TIMIT using Lasagne.")
        print("Usage: %s [paramFile [EPOCHS]]" % sys.argv[0])
        print()
        print("paramFile: the file of well trained parameters")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        for i in range(10):
			for weight in [[1,5],[1,6],[1,7],[1,8]]:
				kwargs['val_data'] = i
				kwargs['weight'] = weight
				print 'weight =',weight
				if len(sys.argv) > 1:
					kwargs['paramFile'] = sys.argv[1]
				if len(sys.argv) > 2:
					kwargs['num_epochs'] = int(sys.argv[2])
				main(**kwargs)




