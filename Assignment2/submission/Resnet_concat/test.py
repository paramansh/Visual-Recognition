import torch
from model_fine import Model
from torch.utils.data import DataLoader
from feature_extractor import get_vector
import argparse
import os
from model_coarse import Model as Model_coarse

parser = argparse.ArgumentParser()
parser.add_argument("--tf",help="takes path to folder containing test images")
args=parser.parse_args()
if(type(args.tf)!=str):
    print("Please enter test_folder with flag --tf")
    exit(0)

files = os.listdir(args.tf)
cnt=0
for file in files:
    cnt+=1
# container=torch.zeros((cnt,512))
dictionary={}
cnt=0
for file in files:
    filename=args.tf+str(file)
    # print(file)
    v=get_vector(filename)
    # dictionary[str(file)]=v
    # v=torch.cat((v,category))
    # container[cnt]=v
    dictionary[str(file)]=v
    cnt+=1
# dictionary['images']=files
# dictionary['test']=container
# valid_data=torch.load("dogs_.pt")
# testloader=DataLoader(dictionary,batch_size=4,shuffle=True)
# dataiter=iter(testloader)
# batch=dataiter.next()
model = Model()
model_coarse = Model_coarse()
model_coarse.load_state_dict(torch.load('model_coarse.pt'))
model_coarse.eval()
# print("Hi")
model.load_state_dict(torch.load('model_fine_concatenated.pt'))
model.eval()
for key in dictionary.keys():
    X=dictionary[key]
    X=X.unsqueeze(1).permute(1,0)

    output_coarse=model_coarse(X)
    c=torch.zeros((X.shape[0],5))
    _, pred=torch.max(output_coarse, 1)
    # print(pred)
    for  i,j in enumerate(pred):
        c[i][j]=1

    # print(X.size())
    # print(c)
    input=torch.cat((X,c),1)
    # print(input)
    outputs=model(input)
    # print(outputs)
    # print(outputs.size())
    _, predicted = torch.max(outputs, 1)
    print("Prediction for "+key+" is class "+str(predicted[0].item()))
    # print(predicted)


# print("Yo")
