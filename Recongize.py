import torch
import torch.nn
import torch.nn.functional as F
import sys
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,64,3,padding=1)
        self.maxpool1=torch.nn.MaxPool2d(2,padding=1)
        self.conv2=torch.nn.Conv2d(64,128,3,padding=1)
        self.maxpool2=torch.nn.MaxPool2d(2,padding=1)
        self.conv3=torch.nn.Conv2d(128,256,3,padding=1)
        # self.conv4=torch.nn.Conv2d(256,256,3,padding=1)
        # self.conv5=torch.nn.Conv2d(256,256,3,padding=1)
        self.maxpool3=torch.nn.MaxPool2d(2,padding=1)
        self.conv6=torch.nn.Conv2d(256,512,3,padding=1)
        # self.conv7 = torch.nn.Conv2d(512, 512, 3,padding=1)
        # self.conv8 = torch.nn.Conv2d(512, 512, 3,padding=1)
        self.maxpool4 = torch.nn.MaxPool2d(4,padding=1)
        # self.conv9 = torch.nn.Conv2d(512, 512, 3,padding=1)
        # self.conv10 = torch.nn.Conv2d(512, 512, 3,padding=1)
        # self.conv11 = torch.nn.Conv2d(512, 512, 3,padding=1)
        # self.maxpool5 = torch.nn.MaxPool2d(2,padding=1)
        self.fulllinear1=torch.nn.Linear(4*4*512,4096)
        # self.fulllinear2 = torch.nn.Linear(2048, 1024)
        self.fulllinear2 = torch.nn.Linear(4096, 2)
        # self.fulllinear3 = torch.nn.Linear(1024, 2)
        # self.dropout=torch.nn.Dropout2d()

    def forward(self,x):
        # print(x.shape)
        x=F.relu(self.maxpool1(self.conv1(x)))
        # print(x.shape)
        x=F.relu(self.maxpool2(self.conv2(x)))
        # print(x.shape)
        # x=F.relu(self.maxpool3(self.conv5(self.conv4(self.conv3(x)))))
        x = F.relu(self.maxpool3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.maxpool4(self.conv6(x)))
        # print(x.shape)
        # x = F.relu(self.maxpool5(self.conv9(x)))
        # print(x.shape)
        # print(x.shape)
        x=x.view(-1,4*4*512)
        # print(x.shape)
        x=self.fulllinear1(x)
        x = F.dropout(x, training=self.training)
        # x=self.fulllinear2(x)
        # x = F.dropout(x,training=self.training)
        x=F.softmax(self.fulllinear2(x))
        # print(x)
        return x

def loadData():
    data=np.load('images.npy')
    labels=np.load('labels.npy')
    labels=np.array(labels,dtype='int')
    piexls_all=data/255.0
    # piexls_all_re=np.reshape(piexls_all.T,(1,784,-1))
    print(piexls_all.shape)
    samples,_,_=piexls_all.shape
    # batchs=np.split(piexls_all,samples//50)
    index_all=np.arange(samples)
    np.random.shuffle(index_all)
    piexls_all=piexls_all[index_all,:,:]
    labels=labels[index_all]
    images_all=np.reshape(piexls_all,(-1,1,128,128))
    print('over loaddata')
    # print(images_all.shape)
    # plt.imshow(images_all[1,:,:,0],cmap='gray_r')
    # plt.show()
    labels_all=np.zeros((samples,2))
    for i in range(len(labels)):
        labels_all[i,labels[i]-1]=1
    # label_batchs=np.split(labels_all,samples//50)
    return images_all,labels_all

if __name__=="__main__":
    train_data,labels=loadData()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data=torch.from_numpy(train_data).to(device=device,dtype=torch.float)
    labels=torch.from_numpy(labels).to(device=device,dtype=torch.float)
    little_bias=torch.from_numpy(np.array([1e-6 for i in range(100)])).to(device=device,dtype=torch.float)
    mynn=NN().to(device)
    opt=optim.SGD(mynn.parameters(),lr=0.0005)
    index_all=np.arange(1500)
    # plt.figure(1)
    # plt.ion()
    # plt.xticks(np.arange(0,200))
    LOSS=[]
    for i in range(20000):
        np.random.shuffle(index_all)
        index_random = index_all[:50]
        mynn.train()
        opt.zero_grad()
        y_predict=mynn(train_data[index_random])
        loss=-torch.sum(torch.log(y_predict.view(1,-1)+little_bias)*labels[index_random].view(1,-1))
        # loss=F.binary_cross_entropy(y_predict.view(1,-1),labels[index_random].view(1,-1))
        loss.backward()
        opt.step()
        # print('loss:', loss.item())
        if i%100==0:
            # try:
            #     plt.cla()
            # except:pass
            LOSS.append(loss.item())
            # plt.plot(LOSS)
            # plt.pause(0.1)
            with torch.no_grad():
                print('loss:', loss.item())
                mynn.eval()
                y_pre=torch.argmax(mynn(train_data[1500:]),dim=1)
                y_real=torch.argmax(labels[1500:],dim=1)
                print('accaurcy:',
                      torch.mean(torch.tensor(y_pre==y_real,dtype=torch.float)))

'''
    test = np.loadtxt('../tensorflow/test.csv', delimiter=',', skiprows=1)
    test_data = test[:, :] / 255.0
    test_samples, _ = test_data.shape
    y_pre=None
    for i in range(test_samples//100):
        images_test = np.reshape(test_data[i*100:i*100+100], (-1, 1, 28, 28)).astype(np.float32)
        mynn.eval()
        y_temp = torch.argmax(mynn(torch.from_numpy(images_test).to(device=device,dtype=torch.float)), dim=1)
        try:
            y_pre = np.concatenate((y_pre,np.reshape(y_temp.cpu().numpy(), (len(y_temp), 1))))
        except:
            y_pre=np.reshape(y_temp.cpu().numpy(), (len(y_temp), 1))
    ImageID = np.arange(1, len(y_pre) + 1).reshape((len(y_pre), 1))
    result = np.concatenate((ImageID, y_pre), axis=1)
    np.savetxt('submisssion.csv', result, fmt='%d', delimiter=',', header="ImageID,Label")
'''





