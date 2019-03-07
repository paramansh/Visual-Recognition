import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
# import yajl

# import copy


# pic_one = str(input("Input first image name\n"))
# pic_two = str(input("Input second image name\n"))

model = models.resnet18(pretrained=True)

layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
classes=["aircrafts","birds_","cars","dogs_","flowers_"]
def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector

    return my_embedding

if __name__ == '__main__' :
    path1 = "/home/subh/CS783/coarse_grained/valid/"
    folders = os.listdir(path1)
    contain=torch.empty((0,0))
    flag=0

    for folder in folders:
        # print(folder)
        # exit(0)
        # if folder=="dogs_":

        print(str(folder))
        path2 = path1+str(folder)+"/"
        files = os.listdir(path2)
        # print(files)
        # exit(0)

        cnt=0

        for file in files:
            cnt+=1
        container=torch.zeros((cnt,517))
        category=torch.zeros((5,))
        index=classes.index(str(folder))
        category[index]=1
        cnt=0
        for file in files:
            filename=path2+str(file)
            v=get_vector(filename)
            v=torch.cat((v,category))
            container[cnt]=v
            cnt+=1
            # print(v.size())
            # exit(0)
            # print(tcontainer.size())
        if(flag==0):
            contain=container
            print("====>first time")
            print(contain.size())
            flag=1
        else:
            contain=torch.cat((contain,container),0)
            print(contain.size())
        # dictionary[str(folder)]=tcontainer
        # print(dictionary[str(folder)].size())
        # exit(0)
    # dictionary["dataset"]=contain
    # print(dictionary["dataset"])
    # print(dictionary["dataset"].size())
    # with open("features.json", 'w') as file:
    #      yajl.dump(dictionary,file)
    torch.save(contain,"valid_less.pt")
# pic_one_vector = get_vector(pic_one)
# pic_two_vector = get_vector(pic_two)
#
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# cos_sim = cos(pic_one_vector.unsqueeze(0),
#               pic_two_vector.unsqueeze(0))
# print('\nCosine similarity: {0}\n'.format(cos_sim))
