import torch
from model import Model
from torch.utils.data import DataLoader
from feature_extractor import get_vector
import argparse
import os

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

model = Model()
# print("Hi")
model.load_state_dict(torch.load('model.pt'))
model.eval()
outlist = []

f0 = open('coarse_output.txt', 'a')
f1 = open('image_list.txt', 'a')
for key in dictionary.keys():
    X=dictionary[key]
    X=X.unsqueeze(1).permute(1,0)
    # print(X.size())
    outputs=model(X)
    _, predicted = torch.max(outputs, 1)
    f0.write(str(predicted[0].item()) + '\n')
    f1.write(key + '\n')  

        # print("Prediction for "+key+" is class "+str(predicted[0].item()))
    # print(predicted)
f0.close()
f1.close()

# print (outlist)

# print("Yo")
