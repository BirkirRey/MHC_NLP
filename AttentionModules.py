import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
from scipy.stats import binom_test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from Bio import pairwise2
from Bio import Entrez
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########The Seq2seq model with attention mechanism is from a GitHub Repository of user AuCson
#https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq

def removeAlleles(df,alleleCol='Allele',removeAlleles=['HLA-A*30:14L','HLA-B*40:01','HLA-B*15:01','HLA-C*15:02']):
    return df[list(map(lambda x:not x in removeAlleles,df[alleleCol].values))]

def getSeqDict(direct, filename,):
    #Take fasta file of HLA seqs and return dicitonary with list of seq records
    #Key to dictionary is an allele name an reclist contains all recs under that allele
    seqDict = {}
    with open(os.path.join(direct,filename),'r') as fh:
        for rec in SeqIO.parse(fh,'fasta'):
            seqDict[rec.description] = rec.seq
    return seqDict

def getSeqNameDict(seqDict,allAlleles):
    seqNameDict = {}
    for seqName in seqDict.keys():
        seqNameProc = seqName.replace('A','HLA-A*')
        seqNameProc = seqNameProc.replace('B','HLA-B*')
        seqNameProc = seqNameProc.replace('C','HLA-C*')
        seqNameProc = seqNameProc[:-2]+':'+seqNameProc[-2:]
        if seqNameProc in allAlleles:
            seqNameDict[seqNameProc] = seqName
    return seqNameDict

def getSeqNameDict2(seqDict,allAlleles):
    seqNameDict = {}
    for seqName in seqDict.keys():
        seqNameProc = seqName.replace('HLA-A','HLA-A*')
        seqNameProc = seqNameProc.replace('HLA-B','HLA-B*')
        seqNameProc = seqNameProc.replace('HLA-C','HLA-C*')
        seqNameProc = seqNameProc[:-2]+':'+seqNameProc[-2:]
        if seqNameProc in allAlleles:
            seqNameDict[seqNameProc] = seqName
    return seqNameDict

def getAllele2SeqDict(allAlleles,seqDict,seqNameDict):
    allele2seqDict={}
    for allele in allAlleles:
        allele2seqDict[allele]=str(seqDict[seqNameDict[allele]])#.replace('-','G')
    return allele2seqDict

def allele2SeqDictWrapper(datDir,filename,allAlleles):
    seqDict = getSeqDict(datDir,filename)
    seqNameDict = getSeqNameDict(seqDict,allAlleles)
    return getAllele2SeqDict(allAlleles,seqDict,seqNameDict)

def allele2SeqDictWrapper2(datDir,filename,allAlleles):
    seqDict = PSEUDO_func(filename)
    seqNameDict = getSeqNameDict2(seqDict,allAlleles)
    return getAllele2SeqDict(allAlleles,seqDict,seqNameDict)

def PSEUDO_func(pseudofile):
    pseudoseq_dict = {}#Dictionary linking allele name to sequence
    for line in open(pseudofile, 'r'):
        key_val = [i.strip() for i in line.split()]
        pseudoseq_dict[key_val[0]]=key_val[1]
        pseudoseq_dict[key_val[1]]=key_val[0]
    return pseudoseq_dict

def ligAllele2ligReceptor(df):
    df['Receptor'] = list(map(lambda allele:allele2SeqDict[allele],df['Allele'].values))
    return df[['Peptide','Receptor']]


def selectXnumOfClass(df,alleleCol,alleles,ligNum=False):
    df = df.sample(frac=1.0)
    df_list = []
    for group in df.groupby(alleleCol):
        if not group[0] in alleles:
            continue
        if ligNum:
            df_list.append(group[1].sample(n=ligNum,replace=True))
        else:
            df_list.append(group[1])
    df_out =  pd.concat(df_list)
    return df_out.sample(frac=1.0)

def makeAA2IDX(df):
    return {aa:i for i,aa in enumerate(df.loc[:,0].values)}

def indexesFromSeq(seq,AA2IDX):
    return [AA2IDX[AA] for AA in seq]

def tensorFromSeq(seq,AA2IDX):
    indexes = indexesFromSeq(seq,AA2IDX)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(pair,AA2IDX):
    input_tensor = tensorFromSeq(pair[0],AA2IDX)
    target_tensor = tensorFromSeq(pair[1],AA2IDX)
    return (input_tensor,target_tensor)

class EncoderRNN2(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.15):
        super(EncoderRNN2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size)
        self.embedding.weight.data.copy_(embeddingTensor)
        #self.embedding.weight.detach_()
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        #outputs, hidden = self.gru(packed, hidden)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

class DynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data.copy_(embeddingTensor)
        #self.embedding.weight.detach_()
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)

        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden, attn_weights


teacher_forcing_ratio = 1.0

def train2(input_tensor,target_tensor, encoder, decoder, encoder_optimizer,
         decoder_optimizer,criterion,max_length=MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
    
    loss = 0
    
    encoder_outputs, encoder_hidden = encoder(input_tensor,[input_length])
    
    decoder_input = torch.tensor([[SOS_token]],device=device)
    
    encoder_hidden=torch.sum(encoder_hidden[0],dim=0)
    encoder_hidden = encoder_hidden.view(1,1,-1)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        #Teacher forcing: Feed the Target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] #Teacher forcing
    else:
        #Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topc, topi = decoder_output.topk(1)
            decoder_input = topi
            #decoder_input = topi.squeeze().detach() #detach from history as input
            
            loss+=criterion(decoder_output,target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length


import time
import math

def asMinutes(s):
    m = math.floor(s/60)
    s-=m*60
    return "%dm %ds"%(m,s)

def timeSince(since, percent):
    now = time.time()
    s=now-since
    es = s/(percent)
    rs = es-s
    return "%s (- %s)"%(asMinutes(s), asMinutes(rs))

def trainIters2(encoder, decoder, n_iters, print_every=1000, plot_every=100,learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print every
    plot_loss_total = 0 # Reset every plot every
    
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate,weight_decay=1e-7)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate,weight_decay=1e-7)
    
    trainIdent=[]
    testIdent=[]
    corr = []
    
    training_pairs = [tensorsFromPair(random.choice(pairs),AA2IDX)
                     for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(1,n_iters+1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train2(input_tensor, target_tensor,encoder,
                     decoder,encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total +=loss
        plot_loss_total +=loss
        
        if iter%print_every ==0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print("%s (%d %d%%) %.4f"%(timeSince(start, iter/n_iters),
                                      iter,iter/n_iters*100,
                                      print_loss_avg))
        if iter % plot_every==0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            correl = sumAttentions_train(encoder, decoder)
            corr.append(correl)
            trainIdent.append(evaluateSeqAcc(encoder,decoder,n=10))
            testIdent.append(evaluateSeqAcc(encoder, decoder, n=10,test=True))
            if abs(correl) > 0.45:
                print(training_pairs)
                return plot_losses,trainIdent,testIdent,corr
            
            
        #if iter % 10:
    showPlot(plot_losses)
    return plot_losses,trainIdent,testIdent,corr


def sumAttentions_train(encoder, decoder, n=3):
    atts = []
    for i in range(n):
        pair = random.choice(pairs)
        output_words,attentions = evaluate2(encoder,decoder,pair[0])
        #print(attentions.shape)
        if attentions.size()[0]==10:#MAX_LENGTH:
            atts.append(attentions)
    if len(atts)==0:
        return 0
    attMean= torch.stack(atts).mean(dim=0).numpy()
    attMean_slice = attMean[:-1,:-1]
    #attMean_slice = attMean[:,:9].T
    #prox = [7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    #prox = list(map(lambda x:x,prox))
    #attMean_slice_proxy = attMean_slice[:,prox]
    plt.matshow(attMean)
    plt.colorbar()
    plt.show()
    plt.matshow(attMean_slice)
    plt.colorbar()
    plt.show()
    attVect = np.reshape(attMean_slice,(1,-1))
    minDist = distMat_proxy.min(axis=0)
    distMat_proxy_rel = distMat_proxy/minDist
    distVect = np.reshape(distMat_proxy,(1,-1))
    #plt.scatter(distVect,attVect)
    corr = np.corrcoef(attVect,distVect)[0,1]
    print(pearsonr(attVect[0],distVect[0]))
    #print(corr)
    return corr

def showPlot(points):
    %matplotlib inline
    plt.figure()
    fig, ax = plt.subplots()
    #this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

    #plt.show()

def evaluate2(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSeq(sentence, AA2IDX)
        input_length = input_tensor.size()[0]
        #encoder_hidden = encoder.initHidden()
        #print(sentence)
        
        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size,device=device)
        
        input_length = input_tensor.size(0)
        #target_length = target_tensor.size(0)
        
        encoder_outputs, encoder_hidden = encoder(input_tensor,[input_length])
        
        
        decoder_input = torch.tensor([[SOS_token]],device=device)
        encoder_hidden=torch.sum(encoder_hidden[0],dim=0)
        encoder_hidden = encoder_hidden.view(1,1,-1)
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        #decoder_attentions = torch.zeros(10, max_length)
        #print(decoder_attentions.shape)
        attList = []
        for di in range(max_length):
            #print('Decoder input')
            #print(decoder_input)
            decoder_output,decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #print('Attention shape')
            #print(decoder_attention.data.shape)
            
            #decoder_attentions[di] = decoder_attention.data
            attList.append(decoder_attention.data)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(IDX2AA[topi.item()])
            #print(topi)
            decoder_input = topi
            #decoder_input = topi.squeeze().detach()
            #decoder_input = torch.tensor([[decoder_input.item()]],device=device)
            #print(decoder_input.shape)
        #print(di) 
        return decoded_words, torch.stack(attList).squeeze()#decoder_attentions[:di+1]

def evaluateRandomly(encoder, decoder, n=10,test=False):
    targPred=[]
    for i in range(n):
        if test:
            pair = random.choice(pairs2)
        else:
            pair = random.choice(pairs2)
        output_words,attentions = evaluate2(encoder,decoder,pair[0])
        output_sentence = ''.join(output_words)
        if output_sentence.endswith('<EOS>'):
            targPred.append((pair[1],output_sentence[:-5]))
        else:
            targPred.append((pair[1],output_sentence))
    return targPred

def evaluateRandomly2(encoder, decoder, n=10,test=False):
    targPred=[]
    for i in range(n):
        if test:
            pair = random.choice(pairs2)
        else:
            pair = random.choice(pairs2)
        print('>',pair[0])
        print('=',pair[1])
        output_words,attentions = evaluate2(encoder,decoder,pair[0])
        output_sentence = ''.join(output_words)
        if output_sentence.endswith('<EOS>'):
            targPred.append((pair[1],output_sentence[:-5]))
        else:
            targPred.append((pair[1],output_sentence))
        print('<',output_sentence)
        print('')
    return targPred

def evaluateRandomly_att(encoder, decoder, n=10):
    atts = []
    for i in range(n):
        pair = random.choice(pairs)
        output_words,attentions = evaluate2(encoder,decoder,pair[0])
        atts.append(attentions)
    return atts

def evaluateRandomly_Single_att_pair(encoder, decoder, n=1):
    atts = []
    for i in range(n):
        pair = random.choice(pairs)
        output_words,attentions = evaluate2(encoder,decoder,pair[0])
        atts.append(attentions)
    return atts, pair,output_words

def seqAccuracy(targ,pred):
    predLen = len(pred)
    errInd = []
    corrCount = 0.0
    if len(targ)!=len(pred):
        return 0.0, [0]
    for i,AA in enumerate(pred):
        if targ[i]==AA:
            corrCount+=1
        else:
            errInd.append(i)
    return corrCount/predLen, errInd

def evaluateSeqAcc(encoder,decoder,n=10,test=False):
    seqAcc = [seqAccuracy(targ,pred)[0] for targ,pred in evaluateRandomly(encoder,decoder,n=n,test=test)]
    return np.array(seqAcc).mean()


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='Blues')
    #fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + list(input_sentence) +
                       ['<EOS>'], rotation=90)
    ax.set_xticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateRandomly_Single_att_pair_out(encoder, decoder, n=1):    
    pair = random.choice(pairs)
    output_words,attentions = evaluate2(encoder,decoder,pair[0])
    print('Input',pair[0])
    print('Output ',output_words)
    return attentions, pair,output_words
    
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
    




