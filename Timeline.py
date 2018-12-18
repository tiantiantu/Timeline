import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

def preprocessing(visitfile,labelfile,gapfile,batchsize):
    visitset=np.load(visitfile)
    labelset=np.load(labelfile)
    gapset=np.load(gapfile)
    

    data=[]
    for i in range(0,len(visitset)):
        data.append((visitset[i], labelset[i], gapset[i]))
        
    data=np.array(data)   
    
    code_to_ix = {}
    label_to_ix={}
    ix_to_code={}
    ix_to_code[0]='OUT'
    
    for visits, label, gap in data:
        for visit in visits:
            for code in visit:
                if code not in code_to_ix:
                    code_to_ix[code] = len(code_to_ix)+1
                    ix_to_code[code_to_ix[code]]=code  
        
        if label not in label_to_ix:
            label_to_ix[label]=len(label_to_ix)
    
    
      
    
    lenlist=[]
    for i in data:
        lenlist.append(len(i[0]))
    sortlen=sorted(range(len(lenlist)), key=lambda k: lenlist[k])  
    new_data=data[sortlen]
    
    
    
    
    batch_data=[]
    
    for start_ix in range(0, len(new_data)-batchsize+1, batchsize):
        thisblock=new_data[start_ix:start_ix+batchsize]
        mybsize= len(thisblock)
        mynumvisit=np.max([len(ii[0]) for ii in thisblock])
        mynumcode=np.max([len(jj) for ii in thisblock for jj in ii[0] ])
        main_matrix = np.zeros((mybsize, mynumvisit, mynumcode), dtype= np.int)
        mask_matrix = np.zeros((mybsize, mynumvisit, mynumcode), dtype= np.float32)
        gap_matrix = np.zeros((mybsize, mynumvisit), dtype= np.float32)
        mylabel=[]
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                for k in range(main_matrix.shape[2]):
                    try:
                        main_matrix[i,j,k] = code_to_ix[thisblock[i][0][j][k]]
                        
                    except IndexError:
                        mask_matrix[i,j,k] = 1e+20
                        
        for i in range(gap_matrix.shape[0]):
            mylabel.append(thisblock[i][1])
            for j in range(gap_matrix.shape[1]):
                try:
                    gap_matrix[i,j]=thisblock[i][2][j]
                except IndexError:
                    pass
        batch_data.append(((autograd.Variable(torch.from_numpy(main_matrix)), autograd.Variable(torch.from_numpy(mask_matrix)), autograd.Variable(torch.from_numpy(gap_matrix))),autograd.Variable(torch.LongTensor(mylabel))))
    
    print ("The number of batches:", len(batch_data))
    print ("The number of labels:", len(label_to_ix))
    print ("The number of codes:", len(code_to_ix))
    return batch_data, code_to_ix, label_to_ix

######################################################################
# Create the model:


class Timeline(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, attention_dim, vocab_size, labelset_size,dropoutrate):
        super(Timeline, self).__init__()
        self.hidden_dim = hidden_dim
        self.batchsi=batch_size
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, labelset_size)
        self.hidden = self.init_hidden()
        self.attention=nn.Linear(embedding_dim, attention_dim)
        self.vector1=nn.Parameter(torch.randn(attention_dim,1))
        self.decay=nn.Parameter(torch.FloatTensor([-0.1]*(vocab_size+1)))    
        self.initial=nn.Parameter(torch.FloatTensor([1.0]*(vocab_size+1)))   
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.attention_dimensionality=attention_dim
        self.WQ1=nn.Linear(embedding_dim, attention_dim,bias=False)
        self.WK1=nn.Linear(embedding_dim, attention_dim,bias=False)
        self.embed_drop = nn.Dropout(p=dropoutrate)
        

    def init_hidden(self):
        
        return (autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim).cuda()))

    def forward(self, sentence, Mode):
        numcode=sentence[0].size()[2]
        numvisit=sentence[0].size()[1]
        numbatch=sentence[0].size()[0]
        thisembeddings =self.word_embeddings(sentence[0].view(-1,numcode))
        thisembeddings = self.embed_drop(thisembeddings)
        myQ1=self.WQ1(thisembeddings)
        myK1=self.WK1(thisembeddings)
        dproduct1= torch.bmm(myQ1, torch.transpose(myK1,1,2)).view(numbatch,numvisit,numcode,numcode)
        dproduct1=dproduct1-sentence[1].view(numbatch,numvisit,1,numcode)-sentence[1].view(numbatch,numvisit,numcode,1)
        sproduct1=self.softmax(dproduct1.view(-1,numcode)/np.sqrt(self.attention_dimensionality)).view(-1,numcode,numcode) 
        fembedding11=torch.bmm(sproduct1,thisembeddings)
        fembedding11=(((sentence[1]-(1e+20))/(-1e+20)).view(-1,numcode,1)*fembedding11)
        mydecay = self.decay[sentence[0].view(-1)].view(numvisit*numbatch,numcode,1)
        myini = self.initial[sentence[0].view(-1)].view(numvisit*numbatch, numcode,1)
        temp1= torch.bmm( mydecay, sentence[2].view(-1,1,1))
        temp2 = self.sigmoid(temp1+myini)   
        vv=torch.bmm(temp2.view(-1,1,numcode),fembedding11)
        vv=vv.view(numbatch,numvisit,-1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(vv, self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = self.softmax(label_space)
        return label_scores

######################################################################
# Train the model:

def train_model(batch_data, val_data, code_to_ix,label_to_ix, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, EPOCH, batchsize,dropoutrate):


    model = Timeline(batchsize, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, len(code_to_ix), len(label_to_ix),dropoutrate)
    model.cuda()
    loss_function = nn.NLLLoss()
    
    optimizer = optim.Adam(model.parameters())
    
    ep=0
    while ep <EPOCH:  
        model.train()
        for mysentence in batch_data:
            model.zero_grad()
            model.hidden = model.init_hidden()
            targets = mysentence[1].cuda()
            label_scores = model((mysentence[0][0].cuda(),mysentence[0][1].cuda(),mysentence[0][2].cuda()), 1)
            loss = loss_function(label_scores, targets)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'model_epoch_'+str(ep))         
        print ('finished', ep, 'epochs')
        print ('on validation set:')
        model.eval()
        model.hidden = model.init_hidden()
    
        y_true=[]
        y_pred=[]
        for inputs in val_data:
            model.hidden = model.init_hidden()  
            tag_scores = model((inputs[0][0].cuda(),inputs[0][1].cuda(),inputs[0][2].cuda()), 1).data
            for sindex in range(0,len(tag_scores)):
                
                y_true.append(inputs[1].data[sindex])
                y_pred.append(torch.max(tag_scores[sindex],-1)[1][0])
     
        print ('accuracy:', f1_score(y_true, y_pred, average='micro')  )
        print ('f1 score:', f1_score(y_true, y_pred, average='weighted'))
        
        
        
        ep=ep+1
    print ("training done")
    #return model
    
def test_model(batch_data, model): 
        model.eval()      
        model.hidden = model.init_hidden()
        
        y_true=[]
        y_pred=[]
        for inputs in batch_data:
            
            model.hidden = model.init_hidden() 
            label_scores = model((inputs[0][0].cuda(),inputs[0][1].cuda(),inputs[0][2].cuda()), 1).data
            for sindex in range(0,len(label_scores)):
                
                y_true.append(inputs[1].data[sindex])
                y_pred.append(torch.max(label_scores[sindex],-1)[1][0])
     
            
            
        
        print ("accuracy: ", f1_score(y_true, y_pred, average='micro'))  
        print ("f1 score: ", f1_score(y_true, y_pred, average='weighted'))
       
    
    
def parse_arguments(parser):
    parser.add_argument('visitfile', type=str)
    parser.add_argument('labelfile', type=str)
    parser.add_argument('gapfile', type=str)
    parser.add_argument('--EMBEDDING_DIM', type=int, default=80)
    parser.add_argument('--HIDDEN_DIM', type=int, default=80)
    parser.add_argument('--ATTENTION_DIM', type=int, default=80)
    parser.add_argument('--EPOCH', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=48)
    parser.add_argument('--dropoutrate', type=float, default=0.2)
    		
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    print (args)
    data, c2ix, l2ix=preprocessing(args.visitfile,args.labelfile,args.gapfile,args.batchsize)
    
    training_data, test_data = train_test_split(data, test_size=0.3, random_state=2)
    test_data, validation_data = train_test_split(test_data, test_size=0.34, random_state=2)
    
    train_model(training_data, validation_data, c2ix,l2ix, args.EMBEDDING_DIM,args.HIDDEN_DIM, args.ATTENTION_DIM,args.EPOCH, args.batchsize,args.dropoutrate)
    
    epoch=0
    print ("performance on the test set:")
    while epoch < args.EPOCH:
        model = Timeline(args.batchsize, args.EMBEDDING_DIM, args.HIDDEN_DIM, args.ATTENTION_DIM, len(c2ix), len(l2ix),args.dropoutrate)
        model.cuda()
        model.load_state_dict(torch.load('model_epoch_'+str(epoch)))
        print ("model after",str(epoch),"epochs")
        test_model(test_data, model)
        epoch=epoch+1
    