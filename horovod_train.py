import argparse
import os
import re
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
#import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from torch import optim


from torch.utils.data import DataLoader,random_split,distributed,Subset
from torchvision import transforms, utils,datasets

from torchvision.models.segmentation import fcn_resnet50
import horovod.torch as hvd


parser = argparse.ArgumentParser(description='CelebAMask-HQ')


parser.add_argument("--dataset-dir",type = str,required=True,
                    help = "input root directory of Images")
parser.add_argument("--label-dir",type = str, required = True,
                   help = "input lables directory")

parser.add_argument("--batch-size",type = int,default = 4,
                    help = "input batch_size for training (default: 96)" )

parser.add_argument("--epochs",type = int,default = 11,
                    help = "input number of epochs (default: 11)")

parser.add_argument("--test-split-perc",type = float,default = 0.25,
                    help = "input percentage of dataset to split into test data (default: 0.25)")

parser.add_argument("--pytorch-model-dir",type = str,default = "./models/pth/",
                    help = "input directory to save PyTorch model (default: ./models/pth/ )")

parser.add_argument("--pytorch-checkpoint-format", default='checkpoint-{epoch}.pth',
                    help="pytorch model checkpoint file format (default: checkpoint-{epoch}.pth)")

parser.add_argument("--onnx-model-dir",type = str,default = "./models/onnx/",
                    help = "input directory to save onnx model (default: ./models/onnx/ )")

parser.add_argument("--onnx-checkpoint-format", default="checkpoint-onnx-{epoch}.onnx",
                    help="onnx checkpoint file format (default: checkpoint-onnx-{epoch}.onnx )")

parser.add_argument("--seed",type = int, default = 64,
                    help = "random seed (default: 64)")
parser.add_argument("--base-lr",type = float,default = 0.025,
                    help = "base learning rate (default:0.025)")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')



kwargs = {'pin_memory': True}
def cross_entropy2d(predicted,true_labels):
    n, c, h, w = predicted.size()
    predicted = predicted.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    
    true_labels = true_labels.view(-1)
    loss = F.cross_entropy(predicted,true_labels.long(),size_average=True)
    
    return loss , predicted,true_labels

def train(epoch):
    model.train()
    model.cuda()
    train_sampler.set_epoch(epoch)
    running_loss = 0
    
    print_every = 5
    for batch_idx,(images,labels) in enumerate(train_dataloader):
        adjust_learning_rate(epoch,batch_idx)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(images)["out"]
        loss,_,_ = cross_entropy2d(logps, labels[:,0,:,:] * 255.0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if hvd.rank() == 0 and batch_idx % print_every ==0:

            print("\rTrain epoch: {} [{}/{} {:0f}%]\t Loss: {}".format(
                epoch,batch_idx* len(images),len(train_dataset), (100. * batch_idx) / len(train_dataloader),loss.item()))

            to_range = 4 if args.batch_size > 4 else args.batch_size
            for x in range(to_range):
                out = torch.argmax(logps[x].permute(1,2,0),axis=2) 

                cv2.imwrite(str(x).rjust(5,"0")+".png",out.to("cpu").numpy())

def test():
    test_loss = 0
    test_accuracy = 0
    total_count = 0
    model.cuda()
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:

            inputs, labels = inputs.cuda(), labels.cuda()
            logps = model.forward(inputs)["out"]
            loss,one_hot_pred,labels = cross_entropy2d(logps, labels[:,0,:,:] * 255.0)
            
            test_loss += loss.item()
            pred = one_hot_pred.argmax(1)
            test_accuracy += pred.eq(labels.data.view_as(pred)).cpu().float().sum()
            total_count += pred.shape[0]

    test_loss /= total_count
    test_accuracy /= total_count
    test_loss = metric_average(test_loss,"test_loss")
    test_accuracy = metric_average(test_accuracy,"test_accuracy")

    if hvd.rank() == 0:
        print("\n Test set : Average loss: {}, Accuracy: {:.2f}\n".format(
            test_loss.item(),100 * test_accuracy.item()))
    return test_loss


def inference():
    model.eval()
    images,labels = next(iter(test_dataloader))
    shape = list(images.shape)
    input = images[0].reshape(1,shape[1],shape[2],shape[3])
    with torch.no_grad():
        now = datetime.now()
        logps = model.forward(input.cuda()).cpu()
        end = datetime.now()
        print("Total inference time for 1 image in = {} s.".format((end - now).total_seconds()))


def metric_average(val,name):
    val_tensor = torch.tensor(val)
    avg_val = hvd.allreduce(val_tensor.detach().cpu(),name = name)
    return avg_val



def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.pytorch_model_dir + args.pytorch_checkpoint_format.format(epoch = epoch)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


def save_onnx(epoch):
    if hvd.rank() == 0:
        images,_ = next(iter(train_dataloader))
        shape = list(images.shape)
        tensor = images[0].reshape(1,shape[1],shape[2],shape[3])
        model.eval()
        torch.onnx.export(model.cpu(),tensor.cpu(),
                        args.onnx_model_dir + args.onnx_checkpoint_format.format(epoch = epoch),
                        export_params=True,
                        opset_version=11,
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}},
                        )

def adjust_learning_rate(epoch, batch_idx):
    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_dataloader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
    elif epoch < 5:
        lr_adj = 0.1
    elif epoch < 10:
        lr_adj = 1e-2
    elif epoch < 15:
        lr_adj = 1e-3
    else:
        lr_adj = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size()  * lr_adj



        
        
class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.dataset = []
        self.preprocess()
        
        self.num_images = len(self.dataset)
    def preprocess(self):
        
        images = os.listdir(self.img_path)

        
        for img in images:
            img_path = os.path.join(self.img_path,img)
            label_path = os.path.join(self.label_path,img[:-4]+".png")
            self.dataset.append([img_path,label_path])
        return self.dataset
    def __getitem__(self,idx):
        
        img_path, label_path = self.dataset[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        return [self.transform_img(img),self.transform_label(label)]
    
    def __len__(self):
        return self.num_images
    


def makeDir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
            
            

if __name__ == '__main__':
    args = parser.parse_args()

    # learning rate warmup
    warmup_epochs = 5
    
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #Initiazie Horovod
    hvd.init()

    #Pin GPU to be used
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")

    
    transform_img = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ])
    transform_label = transforms.Compose([transforms.ToTensor()])
    
    dataset = CelebAMaskHQ(img_path= args.dataset_dir,
                           label_path= args.label_dir,
                           transform_img=transform_img,
                           transform_label=transform_label
                          )    
    dataset_len = len(dataset)
    test_data_len = int( dataset_len * args.test_split_perc)
    
    train_dataset,test_dataset = random_split(dataset,[dataset_len - test_data_len,test_data_len])


    train_sampler = distributed.DistributedSampler(train_dataset,num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = distributed.DistributedSampler(test_dataset,num_replicas=hvd.size(), rank=hvd.rank())

    train_dataloader = DataLoader(dataset = train_dataset,batch_size= args.batch_size,sampler = train_sampler, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset,batch_size= args.batch_size,sampler = test_sampler, **kwargs)

    
        # If set > 0, will resume training from a given checkpoint.
    import re

    resume_from_epoch = False
    epochs_already_trained_list = []
    all_model_names = os.listdir(args.pytorch_model_dir)
    start_epoch_num = None
    if len(all_model_names)  > 0:
        for model_name in all_model_names:
            model_epoch_number = re.findall(r"checkpoint\-(\d+)\.pth", model_name)
            if len(model_epoch_number) > 0:
                resume_from_epoch =  True
                epochs_already_trained_list.append(int(model_epoch_number[0]))
    start_epoch_num = max(epochs_already_trained_list) if len(epochs_already_trained_list) >0 else 0

    start_epoch_num = hvd.broadcast(torch.tensor(start_epoch_num), root_rank=0,name='resume_from_epoch').item()
        
    model = fcn_resnet50()
    if hvd.rank() == 0:
        # Load the pretrained Densenet model only root will download pretrained weights
        model = fcn_resnet50(pretrained = True)


    # add layers
    classifier = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Dropout(p=0.1, inplace=False),
    nn.Conv2d(512, 19, kernel_size=(1, 1), stride=(1, 1)))

    model.classifier = classifier

    model.cuda()

    optimizer = optim.SGD(model.parameters(),lr = args.base_lr ,momentum=args.momentum)


    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    
    # load latest saved checkpoint
    if resume_from_epoch > 0 and hvd.rank() == 0:
        
        filepath = args.pytorch_checkpoint_format.format(epoch=start_epoch_num)
        filepath = args.pytorch_model_dir + filepath
        print("LOADING MODEL :- "+filepath)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("LOADED MODEL :- "+filepath)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    
    makeDir(args.pytorch_model_dir)
    #create directory to save onnx model
    makeDir(args.onnx_model_dir)
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        test()
        
        if epoch % 1 == 0:
            save_checkpoint(epoch)
            save_onnx(epoch)
#             inference() # to check inference time
