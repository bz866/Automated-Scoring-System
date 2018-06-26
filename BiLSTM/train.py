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
from skll.metrics import kappa
import Model
import random
import numpy as np

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
    scaled= []
    essay_set = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        scaled.append(dict["scaled_score2"])
        labels.append(dict["label"])
        essay_set.append(dict["essay_set"])
    return vectors,scaled, labels,essay_set

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

        vectors, scaled,labels,essay_set = get_batch(next(training_iter)) 
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        scaled = Variable(torch.stack(scaled).squeeze().type('torch.FloatTensor'))
        scaled = scaled.cuda() 
        vectors = Variable(vectors)
        vectors = vectors.cuda()

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(vectors, hidden)
        
        lossy = loss_(output, scaled)
        epoch_loss += lossy.data[0] * batch_size

        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()

        if step % total_batches == 0:
            loss_train = evaluate(model, train_eval_iter,batch_size)
            loss_dev = evaluate(model, dev_iter,batch_size)
            kappa_dev = evaluate_kappa(model, dev_iter,batch_size)
            with open("../data/lstm_mot_50epoch.txt", "a") as myfile:
                myfile.write("Epoch %i; Step %i; Avg Loss %f; Train loss: %f; Dev loss: %f; Dev kappa: %s\n" 
                  %(epoch, step, epoch_loss/total_samples, loss_train, loss_dev, kappa_dev))
            print("Epoch %i; Step %i; Avg Loss %f; Train loss: %f; Dev loss: %f; Dev kappa: %s" 
                  %(epoch, step, epoch_loss/total_samples, loss_train, loss_dev, kappa_dev))
            epoch += 1
            
        if step % 5 == 0:
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
        vectors,scaled, labels, essay_set = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        scaled = Variable(torch.stack(scaled).squeeze().type('torch.FloatTensor'))
        scaled = scaled.cuda() 
        vectors = Variable(vectors)
        vectors = vectors.cuda()

        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)
        evalloss += F.mse_loss(output, scaled).data[0]
    return evalloss/len(data_iter)


def evaluate_kappa(model, data_iter, batch_size):
    scale_map = {1:1, 2:4/5, 3:3/10, 4:3/10, 5:1/2.5, 6:1/2.5, 7:3, 8: 6}
    bias_map = {1:+2, 2:+2, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    
    model.eval()
    predicted_labels = []
    true_labels = []
    essay_sets = []
    kappa_list = []
    hidden = model.init_hidden(batch_size)
    for i in range(len(data_iter)):
        vectors, scaled, labels,essay_set = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)

        vectors = vectors.cuda()
        vectors = Variable(vectors)
        
        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)

        predicted = [float(num) for num in output.data.cpu().numpy()]
        de_scale = [(p*scale_map[i])+bias_map[i] for p,i in zip(predicted,essay_set)]


        predicted_labels.extend(de_scale)
        
        labels = torch.stack(labels).squeeze().cpu().numpy()
        true_labels.extend(labels)
        essay_sets.extend(essay_set)
    result = pd.DataFrame({'true':true_labels, 'pred':predicted_labels, 'essay_set':essay_sets})
    for i in range(1,9):
        true_i = result[result.essay_set==i].true
        pred_i = result[result.essay_set==i].pred
        kappa_list.append(kappa(true_i, pred_i, weights = "quadratic"))
    return kappa_list


raw_data = pd.read_csv("../data/training_final_train_val.csv", sep=',',header=0, index_col=0)
data_set = data.get_data(raw_data)
print('Finished Loading!')

#get max sequence length
max_seq_length = max(list(map(lambda x:len(x.split()),raw_data.essay)))
print('max seq length: ', max_seq_length)

# split to train/val/test
data_size = len(data_set)
print('data_size',data_size)
training_set = data_set[:int(data_size*0.8)]
dev_set = data_set[int(data_size*0.8):]


# convert and formatting
word_to_ix, index_to_word, vocab_size = data.build_dictionary([training_set])
#print('vocab size', vocab_size)
data.sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set], max_seq_length)
print('Finished Converting!')

with open('../result/word_to_ix.pk', 'wb') as handle:
    pickle.dump(word_to_ix, handle)

#######
# Train

# Hyper Parameters 
model = 'LSTM'
input_size = vocab_size
hidden_dim = 24
embedding_dim = 50
batch_size = 20
learning_rate = 0.0005
num_epochs = 50
num_layer = 1
bi_direction = True




################################################
# pretrain loading     
# filter GloVe, only keep embeddings in the vocabulary

matrix = np.zeros((2, int(embedding_dim)))
glove = {}
filtered_glove = {}
glove_path = '../data/filtered_glove_50.p'

# reuse pickle file
if(os.path.isfile(glove_path)):
    print("Reusing glove dictionary to save time")
    pretrained_embedding = pickle.load(open(glove_path,'rb'))
else:
    with open('../data/glove.6B.50d.txt') as f:
        lines = f.readlines()
        for l in lines:
            vec = l.split(' ')
            glove[vec[0].lower()] = np.array(vec[1:])
    print('glove size={}'.format(len(glove)))
    print("Finished making glove dictionary")
    # search in vocabulary
    for i in range(2, len(index_to_word)):
        word = index_to_word[i]
        if(word in glove):
            vec = glove[word]
            filtered_glove[word] = glove[word]
            matrix = np.vstack((matrix,vec))
        else:
            # Random initialize
            random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,embedding_dim))
            matrix = np.vstack((matrix,random_init))
            
    pickle.dump(matrix, open("../data/filtered_glove_50.p", "wb"))
    pretrained_embedding = matrix
    print("Saving glove vectors")





# Build, initialize, and train model
rnn = Model.LSTM(model, vocab_size, embedding_dim, hidden_dim, num_layer, dropout=0.2, bidirectional=bi_direction, 
pre_emb=pretrained_embedding)
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

torch.save(rnn, '../result/lstm-mot-50epoch.model')
