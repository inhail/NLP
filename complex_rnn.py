'''
source: https://gist.github.com/karpathy/d4dee566867f8291f086
https://gist.github.com/karpathy/d4dee566867f8291f086#file-min-char-rnn-py-L102
https://www.youtube.com/watch?v=cO0a0QYmFm8&feature=youtu.be&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&t=836
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
'''

import numpy as np

# data I/O
data = open('input.txt','r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' %(data_size,vocab_size)) 
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_layers=100
seq_length= 25 # each training step, we take 25 char into consideration
learning_rate= 1e-3

# model parameters
Wxh = np.random.randn(hidden_layers,vocab_size)*0.001 # weight of 'input to hidden layer'
Whh = np.random.randn(hidden_layers,hidden_layers)*0.001 # weight of 'hidden to hidden'
Why = np.random.randn(vocab_size,hidden_layers)*0.001 # weight of 'hidden to output'
bh = np.zeros((hidden_layers,1)) # hidden bias
by = np.zeros((vocab_size,1)) # output bias
# both bh and by are 2D array with (hidden_layer,1) shape

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs,ys,ps = {},{},{},{} # 's' = state
    hs[-1]=np.copy(hprev)
    loss=0

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        ''' suppose vocab_size=4
        xs = [ [0,0,0,0],    -> first char state
               [0,0,0,0],    -> second char state
               [0,0,0,0],
               [0,0,0,0] ]
        for each char state, [a,b,c,d], each element inside will renew after next char input
        suppose t=0, we set first element of first char state = 1,that is, [1,0,0,0]
        suppose t=1, we set seconde element of second char state = 1, that is, [?,1,0,0]
        '''
        hs[t] = np.tanh(np.dot(Wxh,xs[t]))+np.dot(Whh,hs[t-1]+bh)
        ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # normalized probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
           dy = np.copy(ps[t])
           dy[targets[t]]-=1 #http://cs231n.github.io/neural-networks-case-study/#grad
           dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
           np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
       '''
       sample a sequence of integers from the model 
       h is memory state, seed_ix is seed letter for first time step
       '''
       x = np.zeros((vocab_size,1)) # vocab_size = len(chars)
       x[seed_ix]=1
       ixes=[]
       for t in range(n):
              h = np.tanh(np.dot(Wxh, x) + np.dot(Whh,h) + bh) # forward pass alike
              y = np.dot(Why, h) + by # forward pass alike
              p = np.exp(y) / np.sum(np.exp(y)) # normalized propability
              ix = np.random.choice(range(vocab_size), p=p.ravel())
              x = np.zeros((vocab_size, 1))
              x[ix] = 1
              ixes.append(ix)
       return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
       # prepare inputs (we're sweeping from left to right in steps seq_length long)
       if p+seq_length+1 >= len(data) or n == 0: 
              hprev = np.zeros((hidden_layers,1)) # reset RNN memory
              p = 0 # go from start of data
              inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
              targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
              
       # sample from the model now and then
       if n % 100 == 0:
              sample_ix = sample(hprev, inputs[0], 200)
              txt = ''.join(ix_to_char[ix] for ix in sample_ix)
              print('----\n %s \n----' % (txt, )) 
                     
       # forward seq_length characters through the net and fetch gradient
       loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
       smooth_loss = smooth_loss * 0.999 + loss * 0.001
       if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress
       
       # perform parameter update with Adagrad
       for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
              mem += dparam * dparam
              param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
       
       p += seq_length # move data pointer
       n += 1 # iteration counter 
