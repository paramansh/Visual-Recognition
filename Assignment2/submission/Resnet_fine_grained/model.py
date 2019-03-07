import torch.nn.functional as F
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1=torch.nn.Linear(512,128)
        self.fc2=torch.nn.Linear(128,36)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x

if __name__ == '__main__' :
    model=Model()
    loss_func=torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-6, momentum = 0.9, nesterov = True)
    train_data=torch.load("train.pt")
    valid_data=torch.load("val.pt")
    trainloader=DataLoader(train_data,batch_size=4,shuffle=True)
    validloader=DataLoader(valid_data,batch_size=4,shuffle=True)
    epochs=[]
    epochs_train_loss=[]
    epochs_val_loss=[]
    for epoch in range(1,151):
        epochs.append(epoch+1)
        train_loss,valid_loss=[],[]
        model.train()
        for data in trainloader:
            # print(data.size())
            # exit(0)
            X=data[:,:512]
            Y=data[:,512:548]
            X, Y = Variable(X), Variable(Y)
            # Y=Y.permute(0,2,1)
            # Y=torch.float(Y)
            # print(Y.dtype)
            # X=Variable(X,requires_grad=True)
            # Y=Y.type(torch.LongTensor)
            # print(Y.dtype)
            # print(Y.size())
            # print(Y)
            # exit(0)
            optimizer.zero_grad()
            output=model(X)
            # print(output.dtype)
            # print(output.size())
            # print(output)
            # print(torch.max(Y, 1))
            # print(torch.max(Y, 1)[1])
            # exit(0)
            loss=loss_func(output,torch.max(Y, 1)[1])
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        model.eval()
        for data in validloader:
            X=data[:,:512]
            Y=data[:,512:548]
            output=model(X)
            loss=loss_func(output,torch.max(Y, 1)[1])
            valid_loss.append(loss.item())
        epochs_train_loss.append(train_loss[-1])
        epochs_val_loss.append(valid_loss[-1])

        print("=======>Training loss after epoch number "+str(epoch)+"is "+str(train_loss[-1]))
        print("=======>Validation loss after epoch number "+str(epoch)+"is "+str(valid_loss[-1]))

    # model.save_state_dict('mytrained.pt')
    print('=========>Done')
    plt.plot(epochs,epochs_train_loss,c='r',label='training loss')
    plt.plot(epochs,epochs_val_loss,c='b',label='validation loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Fine grained Classification")
    plt.show()
    # model.eval()
    # exit(0)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    torch.save(model.state_dict(), 'model_fine.pt')



