from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import pickle
import numpy as np
from keras.layers.merge import concatenate

class dataSet:
    def __init__(self,train=True):
        self.train=train

    def importData(self,dataFile):
        # unpickle the data
        self.data=pickle.load(open(dataFile, 'rb'))

        #initialize with zeros for padding
        self.sen1=np.zeros(self.data.shape[1],questionLen,gloveDim)
        self.sen2=np.zeros(self.data.shape[1],questionLen,gloveDim)
        if train==True:
            self.labels=np.zeros(self.data.shape[1])

        #pick first 30 words of each sentence, assign integers to labels
        for i in range(0,data.shape[1]):
            if len(self.data[i]['sentence1']) < questionLen:
                self.sen1[i,0:len(self.data[i]['sentence1'])] = self.data[i]['sentence1']
            else
                self.sen1[i,:]=self.data[i]['sentence1'][0:questionLen]

            if len(self.data[i]['sentence2']) < questionLen:
                self.sen2[i,0:len(self.data[i]['sentence2'])] = self.data[i]['sentence2']
            else
                self.sen2[i,:]=self.data[i]['sentence2'][0:questionLen]

            if train==True:
                if self.data[i]['gold_label']=='entailment':
                    self.labels[i]=1;

                elif self.data[i]['gold_label']=='contradiction':
                    self.labels[i]=2;
        if train==True:
            self.one_hot=keras.utils.to_categorical(self.labels, num_classes=3)

        data=None

class lstmModel:

    def __init__(self,questionLen=30,gloveDim=100,outLen=100,dropoutRate=0.2,regularize=0.01,activation='tanh'):
        self.questionLen=questionLen
        self.gloveDim=gloveDim
        self.outLen=outLen
        self.dropoutRate=dropoutRate
        self.regularize=regularize
        self.activation=activation
        self.getModel()

    def getModel(self):
        # LSTM layer transforms the two input sentences
        sen1 = Input(shape=(self.questionLen,self.gloveDim))
        sen2 = Input(shape=(self.questionLen,self.gloveDim))
        shared_LSTM=LSTM(outLen)
        encoded1 = shared_LSTM(sen1)
        encoded2 = shared_LSTM(sen2)
        merged = keras.layers.concatenate([encoded1,encoded2],axis=-1)
        x=Dropout(self.dropoutRate)(merged)

        # 3 hidden layers
        x=Dense(2*self.outLen,activation=self.activation,kernel_regularizer=regularizers.l2(self.regularize))(x)
        x=Dropout(self.dropoutRate)(merged)
        x=Dense(2*self.outLen,activation=self.activation,kernel_regularizer=regularizers.l2(self.regularize))(x)
        x=Dropout(self.dropoutRate)(merged)
        x=Dense(2*self.outLen,activation=self.activation,kernel_regularizer=regularizers.l2(self.regularize))(x)
        x=Dropout(self.dropoutRate)(merged)

        # classifier
        predictions=Dense(3,activation='softmax')(x)

        self.model = Model(inputs=[sen1, sen2], outputs=predictions)
        print("Model created")
        return self.model

    def compile():
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    def train(self,dataset,numEpochs,batchSize):
        inputs=[dataset.sen1,dataset.sen2]
        self.model.fit(inputs,dataset.one_hot,epochs=numEpochs,batch_size=batchSize)
        print("Training complete")
