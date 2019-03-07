import os

import sys
import argparse
parser=argparse.ArgumentParser()
# parser.add_argument("--imclass",help="Image Class")
parser.add_argument("--image_file", help="Specify ImageFile to be predicted")
args = parser.parse_args()

from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from config import BATCH_SIZE, PROPOSAL_NUM,test_model
from core import model, dataset
from core.utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if not test_model:
    raise NameError('please set the test_model file to choose the checkpoint!')

test_model = 'fine_grained.ckpt'

# read dataset
# print("Loading Data")
testset = dataset.Predict_data(image_file=args.image_file, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
# print("Data Loaded")
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
ckpt = torch.load(test_model)
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()



net.eval()

test_loss = 0
test_correct = 0
total = 0
out = []
for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        # print(data[1])
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img)
        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        output = concat_predict.data
        # print(concat_loss.item())
        out += list(output.to("cpu").numpy())
        # total += batch_size
        # test_correct += torch.sum(concat_predict.data == label.data)
        # test_loss += concat_loss.item() * batch_size
        # progress_bar(i, len(testloader), 'eval on test set')
# out = out + 1
with open('class_labels.txt') as f:
    class_labels = f.read().splitlines()
class_labels = [i.split(' ')[1] for i in class_labels]
out = [class_labels[int(i)] for i in out]
# print(out)
f3 = open('output.txt', 'a')
img_txt_file = [line.rstrip('\n') for line in open(args.image_file)]
assert (len(out) == len(img_txt_file))

for index, label in enumerate(out):
    if (label[0] == 'a'):
        f3.write(img_txt_file[index] + " " + "aircrafts " + label[1:] + '\n')
    if (label[0] == 'b'):
        f3.write(img_txt_file[index] + " " + "birds " + label[1:] + '\n')
    if (label[0] == 'c'):
        f3.write(img_txt_file[index] + " " + "cars " + label[1:] + '\n')
    if (label[0] == 'd'):
        f3.write(img_txt_file[index] + " " + "dogs " + label[1:] + '\n')
    if (label[0] == 'f'):
        f3.write(img_txt_file[index] + " " + "flowers " + label[1:] + '\n')

