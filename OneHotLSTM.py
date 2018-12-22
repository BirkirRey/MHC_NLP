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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.utils.data

#from sklearn.manifold import TSNE

#The LSTM defined in this module is inspired by the pytorch LSTM tutorial

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

def makeMHC2IDX(df,alleleCol):
    alleles = list(set(df[alleleCol].values))
    alleles.sort()
    return {allele:i for i,allele in enumerate(alleles)}

def makeIDX2MHC(MHC2IDX):
    return {val:key for key,val in MHC2IDX.items()}

def prepare_sequence(seq,toIDX):
    idxs = [toIDX[aa] for aa in seq]
    return torch.tensor(idxs,dtype=torch.long)

def DF2Dat(df):
    return [(list(dat[0]),[dat[1]]) for dat in df.values]

def initDataWrapper(df_data,df_embedding,alleleCol):
    AA2IDX = makeAA2IDX(df_embedding)
    MHC2IDX = makeMHC2IDX(df_data,alleleCol)
    IDX2MHC = makeIDX2MHC(MHC2IDX)
    training_data = DF2Dat(df_data)
    return training_data,AA2IDX,MHC2IDX,IDX2MHC


class lig2allele(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size,BATCH_SIZE,bidirect=False):
        super(lig2allele,self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = bidirect
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(embeddingTensor)
        self.embeddings.weight.detach_()#This is done so that embeddings are not updated
        
        # The LSTM takes word embeddings as input, and outputs hidden states
        #With dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        #The linear layer that maps from hidden state space to tag space
        if self.bidirect:
            self.hidden2tag = nn.Linear(hidden_dim*4,tagset_size)
            self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        else:
            self.hidden2tag = nn.Linear(hidden_dim*2,tagset_size)
            self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        #Before we´ve done anything, we dont have any hidden state.
        #Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        #The axes semantics are (num_layers, minibatch_size,hidden_dim)
        if self.bidirect:
            return (torch.zeros(2,BATCH_SIZE,self.hidden_dim),
                torch.zeros(2,BATCH_SIZE,self.hidden_dim))
        else:
            return (torch.zeros(1,BATCH_SIZE,self.hidden_dim),
                torch.zeros(1,BATCH_SIZE,self.hidden_dim))
    
    def forward(self,sentence):
        embeds = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(sentence.size()[0],BATCH_SIZE,-1), self.hidden)
        #tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))#Original
        x=torch.cat((torch.mean(lstm_out, dim=0), torch.max(lstm_out, dim=0)[0]), dim=1)
        tag_space = self.hidden2tag(x)#Original
        #tag_space = self.activation(tag_space)
        #x = torch.cat((torch.mean(lstm_out, dim=0), torch.max(lstm_out, dim=0)[0]), dim=1)
        #tag_space = self.hidden2tag(lstm_out[len(sentence)-1,:,:])
        
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



def getAccuracy(targPreds):
    boolList = [targ==pred for targ,pred in targPreds]
    return len(boolList), sum(boolList), sum(boolList)/float(len(boolList))

def getTagScores(eval_iter):
    with torch.no_grad():
        tagScores = []
        for sequences,alleles in eval_iter:  
            model.zero_grad()
            model.hidden = model.init_hidden()
            inputs = list(map(lambda x:prepare_sequence(x,AA2IDX),sequences))
            inputs=torch.stack(inputs)
            tag_scores = model(inputs)
            for t,a in zip(tag_scores,alleles[0]):
                tagScores.append((t,a))
    return tagScores

def evaluateModel(eval_iter):
    with torch.no_grad():
        lossList = []
        targPred = []
        for sequences,alleles in eval_iter:  
            model.zero_grad()
            model.hidden = model.init_hidden()
            inputs = list(map(lambda x:prepare_sequence(x,AA2IDX),sequences))
            inputs=torch.stack(inputs)
            tag_scores = model(inputs)
            pred_IDX = torch.argmax(tag_scores,dim=1)
            preds = list(map(lambda x:IDX2MHC[x.item()],list(pred_IDX)))
            targets = list(map(lambda x:prepare_sequence(x,MHC2IDX),alleles))[0]
            lossList.append(loss_function(tag_scores, targets).item())
            #print(preds)
            #print(alleles)
            for t,p in zip(alleles[0],preds):
                targPred.append((t,p))
        meanLoss = np.array(lossList).mean()
    return meanLoss,targPred,getAccuracy(targPred)[2]

def makeMHC2IDX_fly(targPred):
    targ,pred = zip(*targPred)
    targPred = targ+pred
    targPred = list(set(targPred))
    targPred.sort()
    targPred_len = len(targPred)
    return dict(zip(targPred,range(len(targPred))))

def confusionMatrix(targPred,labels=True,titleFont=20,ticker=1,saveFig=False):
    targ,pred = zip(*targPred)
    MHC2IDX_fly = makeMHC2IDX_fly(targPred)
    IDX2MHC_fly = {val:key for key, val in MHC2IDX_fly.items()}
    targ = list(map(lambda x:MHC2IDX_fly[x],targ))
    pred = list(map(lambda x:MHC2IDX_fly[x],pred))
    mattCorr= matthews_corrcoef(targ,pred)
    conMat = confusion_matrix(targ,pred)

    plt.imshow(conMat)
    idx2mhc = list(IDX2MHC_fly.items())
    idx2mhc.sort()
    _,mhcs = zip(*idx2mhc)
    sns.set(font_scale = 2)
    if labels:
        ax = sns.heatmap(conMat,annot=False,fmt='g',
                    xticklabels=mhcs,yticklabels=mhcs,cmap="Blues")
    else:
        ax = sns.heatmap(conMat,annot=False,xticklabels=False,yticklabels=False,fmt='g',cmap="Blues")
    plt.title('MattCorr: %.3f'%mattCorr,fontsize=titleFont)
    for i,label in enumerate(ax.xaxis.get_ticklabels()):
        if i%ticker==0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    for i,label in enumerate(ax.yaxis.get_ticklabels()):
        if i%ticker==0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    if saveFig:
        #plt.tight_layout()
        plt.savefig(saveFig, bbox_inches='tight')
    plt.show()
    return conMat

def targPredCut(targPred,idx=-3):
    return [(targ[:idx],pred[:idx]) for targ,pred in train_targPred]
