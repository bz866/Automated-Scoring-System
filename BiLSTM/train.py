import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os 
import data
import pickle
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import Model
import random


# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def training_loop(batch_size, num_epochs, model, loss_, optim, training_iter, dev_iter, train_eval_iter):
    step = 0
    epoch = 0
    total_batches = int(len(training_set) / batch_size)
    total_samples = total_batches * batch_size
    hidden = model.init_hidden(batch_size)
    while epoch <= num_epochs:
        epoch_loss = 0
        model.train()

        vectors, labels = get_batch(next(training_iter)) 
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        labels = Variable(torch.stack(labels).squeeze().type('torch.FloatTensor'))
        labels = labels.cuda() 
        vectors = Variable(vectors)
        vectors = vectors.cuda()

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(vectors, hidden)
        lossy = loss_(output, labels)
        epoch_loss += lossy.data[0] * batch_size

        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()

        if step % total_batches == 0:
            loss_train = evaluate(model, train_eval_iter,batch_size)
            loss_dev = evaluate(model, dev_iter,batch_size)
            kappa_dev = evaluate_kappa(model, dev_iter,batch_size)
            with open("test.txt", "a") as myfile:
                myfile.write("Epoch %i; Step %i; Avg Loss %f; Train loss: %f; Dev loss: %f; Dev kappa: %f\n" 
                  %(epoch, step, epoch_loss/total_samples, loss_train, loss_dev, kappa_dev))
            print("Epoch %i; Step %i; Avg Loss %f; Train loss: %f; Dev loss: %f; Dev kappa: %f" 
                  %(epoch, step, epoch_loss/total_samples, loss_train, loss_dev, kappa_dev))
            epoch += 1
            
        if step % 5 == 0:
            with open("test.txt", "a") as myfile:
                myfile.write("Epoch %i; Step %i; loss %f\n" %(epoch, step, lossy.data[0]))
            print("Epoch %i; Step %i; loss %f" %(epoch, step, lossy.data[0]))
        step += 1

# This function outputs the accuracy on the dataset, we will use it during training.
def evaluate(model, data_iter, batch_size):
    model.eval()
    correct = 0
    total = 0
    evalloss = 0.0
    hidden = model.init_hidden(batch_size)
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        labels = Variable(torch.stack(labels).squeeze().type('torch.FloatTensor'))
        labels = labels.cuda() 
        vectors = Variable(vectors)
        vectors = vectors.cuda()

        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)
        evalloss += F.mse_loss(output, labels).data[0]
    return evalloss/len(data_iter)


def evaluate_kappa(model, data_iter, batch_size):
    model.eval()
    predicted_labels = []
    true_labels = []
    hidden = model.init_hidden(batch_size)
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)

        vectors = vectors.cuda()
        vectors = Variable(vectors)
        
        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)

        predicted = [int(round(float(num))) for num in output.data.cpu().numpy()]
        predicted_labels.extend([round(float(num)) for num in output.data.cpu().numpy()])
        labels = [int(label[0]) for label in labels]
        true_labels.extend(labels)

    return cohen_kappa_score(true_labels, predicted_labels, weights = "quadratic")


raw_data = pd.read_csv("../data/training_final.csv", sep=',',header=0, index_col=0)
data_set = data.get_data(raw_data)
print('Finished Loading!')

#get max sequence length
max_seq_length = max(list(map(lambda x:len(x.split()),raw_data.essay)))
print('max seq length: ', max_seq_length)

# split to train/val/test
data_size = len(data_set)
print('data_size',data_size)
training_set = data_set[:int(data_size*0.8)]
dev_set = data_set[int(data_size*0.8):int(data_size*0.9)]
test_set = data_set[int(data_size*0.9):]


# convert and formatting
word_to_ix, vocab_size = data.build_dictionary([training_set])
#print('vocab size', vocab_size)
data.sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set], max_seq_length)
print('Finished Converting!')



#######
# Train

# Hyper Parameters 
model = 'LSTM'
input_size = vocab_size
hidden_dim = 24
embedding_dim = 100
batch_size = 20
learning_rate = 0.1
num_epochs = 500
num_layer = 1
bi_direction = True

# Build, initialize, and train model
rnn = Model.LSTM(model, vocab_size, embedding_dim, hidden_dim, num_layer, dropout=0.2, bidirectional=bi_direction, 
pre_emb=embedding_matrix)
rnn.cuda()

# Loss and Optimizer
loss = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Train the model
training_iter = data_iter(training_set, batch_size)
train_eval_iter = eval_iter(training_set, batch_size)
dev_iter = eval_iter(dev_set, batch_size)
print('start training:')
training_loop(batch_size, num_epochs, rnn, loss, optimizer, training_iter, dev_iter, train_eval_iter)

