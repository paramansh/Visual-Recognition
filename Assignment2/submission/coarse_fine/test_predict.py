import os

import sys
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--imclass",help="Image Class")
parser.add_argument("--image", help="Specify Image to be predicted")
args = parser.parse_args()

from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from config import BATCH_SIZE, PROPOSAL_NUM,test_model
from core import model, dataset
from core.utils import progress_bar
# BATCH_SIZE=2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if not test_model:
    raise NameError('please set the test_model file to choose the checkpoint!')

image = args.image
data_dir = args.imclass
test_model = os.path.join('models', args.imclass + '.ckpt')
print("Image Class ", data_dir)

# read dataset
# print("Loading Data")
testset = dataset.PredictClass(imclass=args.imclass, data_len=None)
# testset = dataset.Predict(image=image, data_len=None)
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
    # print("i", i)
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        # print(data[1])
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img)
        # calculate loss
        # concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        output = concat_predict.data
        # print(output.to("cpu").numpy()[0])
        # print(output.to("cpu").numpy())
        out += list(output.to("cpu").numpy())
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)
        # test_loss += concat_loss.item() * batch_size
        # progress_bar(i, len(testloader), 'eval on test set')
print (out)
imclass = args.imclass
f3 = open('output.txt', 'a')
img_txt_file = [line.rstrip('\n') for line in open(os.path.join('coarse_output', imclass + '.txt'))]
assert (len(out) == len(img_txt_file))
for index, o in enumerate(out):
    f3.write(img_txt_file[index] + " " + imclass + " " + str(int(o) + 1)+'\n') # int(o) + 1 in order to align with the actual class number which in our case is just str(model_class) + 1
# test_acc = float(test_correct) / total
# test_loss = test_loss / total
# print('test set loss: {:.3f} and test set acc: {:.3f} total sample: {}'.format(test_loss, test_acc, total))

# print('finishing testing')
