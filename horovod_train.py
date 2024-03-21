import argparse
import os
import re
from PIL import Image
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader,random_split,distributed
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50

import horovod.torch as hvd


parser = argparse.ArgumentParser(description='Training and Evaluating, Segmentation Model on CelebAMask-HQ dataset')


parser.add_argument("--dataset-dir", type = str, required=True,
                    help = "input root directory of Images")
parser.add_argument("--label-dir",type = str, required = True,
                   help = "input lables directory")

parser.add_argument("--batch-size", type = int, default = 4,
                    help = "input batch_size for training (default: 96)" )

parser.add_argument("--epochs",type = int, default = 11,
                    help = "input number of epochs (default: 11)")

parser.add_argument("--test-split-perc", type = float, default = 0.25,
                    help = "input percentage of dataset to split into test data (default: 0.25)")

parser.add_argument("--pytorch-model-dir", type = str, default = "./models/pth/",
                    help = "input directory to save PyTorch model (default: ./models/pth/ )")

parser.add_argument("--pytorch-checkpoint-format", default='checkpoint-{epoch}.pth',
                    help="pytorch model checkpoint file format (default: checkpoint-{epoch}.pth)")

parser.add_argument("--onnx-model-dir", type = str, default = "./models/onnx/",
                    help = "input directory to save onnx model (default: ./models/onnx/ )")

parser.add_argument("--onnx-checkpoint-format", default="checkpoint-onnx-{epoch}.onnx",
                    help="onnx checkpoint file format (default: checkpoint-onnx-{epoch}.onnx )")

parser.add_argument("--seed",type = int, default = 64,
                    help = "random seed (default: 64)")
parser.add_argument("--base-lr", type = float, default = 0.025,
                    help = "base learning rate (default:0.025)")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--resume', action='store_true',
                    help='resume flag starts training from the last epoch it completed')

def makeDir(path):
    """
    Creates a directory at the specified path if it does not already exist.

    Parameters:
    -----------
    path (str): The directory path to be created.
    """

    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


kwargs = {'pin_memory': True}
def cross_entropy2d(predicted,true_labels):
    """
    Computes the cross-entropy loss for 2D inputs.

    Parameters:
    -----------
    predicted (torch.Tensor): The predicted output tensor from the model, with shape 
                              (batch_size, num_classes, height, width).
    true_labels (torch.Tensor): The ground truth labels with shape (batch_size, height, width).

    Returns:
    tuple: A tuple containing:
        - loss (torch.Tensor): The computed cross-entropy loss.
        - predicted (torch.Tensor): The reshaped predicted tensor with shape 
                                    (batch_size * height * width, num_classes).
        - true_labels (torch.Tensor): The reshaped ground truth labels with shape 
                                      (batch_size * height * width).
    """
    _, c, _, _ = predicted.size()
    # this will transfrom from shape (n, c, h, w) to (n, h, w, c) then to (n * h * w, c)
    predicted = predicted.permute(0, 2, 3, 1).contiguous().view(-1, c)
    
    # transfrom from (n, h, w) to (n * h * w) shape
    true_labels = true_labels.view(-1)
    loss = F.cross_entropy(predicted,true_labels.long(),size_average=True)
    
    return loss , predicted,true_labels

def train(epoch, model, train_sampler, train_dataset, optimizer):
    """
    Trains the model for one epoch and prints training progress at specified intervals. It also adjusts 
    the learning rate during training.

    Parameters:
    -----------
    epoch : int
        The current epoch number.
    model : torch.nn.Module
        The model to be trained.
    train_sampler : torch.utils.data.DistributedSampler
        A sampler to handle data shuffling in a distributed training setup.
    train_dataset : torch.utils.data.Dataset
        The dataset containing the training data.
    optimizer : torch.optim.Optimizer
        The optimizer
    """

    model.train()
    model.cuda()
    train_sampler.set_epoch(epoch) # Important for shuffling
    
    running_loss = 0
    progress_print_interval = 5
    for batch_idx,(images,labels) in enumerate(train_dataloader):

        adjust_learning_rate(epoch,batch_idx, optimizer)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(images)["out"]
        loss,_,_ = cross_entropy2d(logps, labels[:,0,:,:] * 255.0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if hvd.rank() == 0 and batch_idx % progress_print_interval ==0:

            print("\rTrain epoch: {} [{}/{} {:0f}%]\t Loss: {}".format(
                epoch,batch_idx* len(images),len(train_dataset), (100. * batch_idx) / len(train_dataloader),loss.item()))


        ################################################################################
                    # code to visualize model is learning well while training
        ################################################################################
        if hvd.rank() == 0 and batch_idx % 100 ==0:
            to_range = 4 if args.batch_size > 4 else args.batch_size
            for x in range(to_range):
                out = torch.argmax(logps[x].permute(1,2,0),axis=2) 
                cv2.imwrite(str(x).rjust(5,"0")+".png",out.to("cpu").numpy())


def test(model, test_dataloader):
    """
    Evaluates the model on the test dataset.

    This function runs the model on the provided test data, computes the loss using 
    a custom cross-entropy function, and calculates the pixel-wise accuracy. The 
    accuracy is calculated as the number of correctly predicted pixels divided by 
    the total number of pixels in the ground truth labels.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model to be evaluated.
    test_dataloader : torch.utils.data.DataLoader
        DataLoader object containing the test dataset.

    Returns:
    --------
    test_loss : float
        The average loss computed over the test set.
    test_accuracy : float
        The pixel accuracy computed over the test set (ratio of correct pixels).
    """

    test_loss = 0
    correct_pixels = 0
    total_pixels = 0
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            logps = model.forward(inputs)["out"]
            
            # Calculating loss
            loss, one_hot_pred, labels = cross_entropy2d(logps, labels[:, 0, :, :] * 255.0)
            test_loss += loss.item()
            
            # Prediction and accuracy calculation
            pred = one_hot_pred.argmax(1)
            correct_pixels += pred.eq(labels).sum().item()
            total_pixels += labels.numel()
            
    test_loss /= len(test_dataloader)
    test_accuracy = correct_pixels / total_pixels  # Pixel accuracy

    # Average metrics across multiple devices
    test_loss = metric_average(test_loss, "test_loss")
    test_accuracy = metric_average(test_accuracy, "test_accuracy")

    if hvd.rank() == 0:
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            test_loss, 100 * test_accuracy))
    
    return test_loss, test_accuracy



def metric_average(val, name):
    """
    Averages a metric value across all processes.

    Parameters:
    -----------
    val : float
        The metric value to be averaged across processes (e.g., loss or accuracy).
    name : str
        A unique name associated with this metric, used by Horovod for identification.

    Returns:
    --------
    float
        The average metric value across all processes.
    """
    # Convert the value to a tensor and move it to GPU if available
    val_tensor = torch.tensor(val, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    # Use Horovod's allreduce to calculate the average value across all processes
    avg_val = hvd.allreduce(val_tensor, name=name)
    return avg_val



def save_checkpoint(epoch, model, optimizer):
    """ 
    This function save pytorch model in the location specified 
    for argument --pytorch-model-dir
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to be exported to onnx format and saved.
    optimizer : torch.optim.Optimizer
        The optimizer to be save in the file

    """
    if hvd.rank() != 0:
        return
    filepath = args.pytorch_model_dir + args.pytorch_checkpoint_format.format(epoch = epoch)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filepath)


def save_onnx(epoch, model):
    """
    Saves the model as an ONNX file at the specified epoch.

    Parameters:
    -----------
    epoch : int
        The current epoch number. This is used to name the ONNX file 
    model : torch.nn.Module
        The model to be exported to onnx format and saved.
    """
    if hvd.rank() != 0:
        return
    
    images,_ = next(iter(train_dataloader))
    shape = list(images.shape)
    tensor = images[0].reshape(1,shape[1],shape[2],shape[3])

    torch.onnx.export(model.eval().cpu(),tensor.cpu(),
                    args.onnx_model_dir + args.onnx_checkpoint_format.format(epoch = epoch),
                    export_params=True,
                    opset_version=11,
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}},
                    )

def adjust_learning_rate(epoch, batch_idx, optimizer):
    """
    Adjusts the learning rate based on the current epoch and batch index.

    The learning rate is adjusted using a warmup strategy during the early epochs
    and then reduced at specific intervals. The adjustment is scaled based on the 
    number of processes in distributed training using Horovod.

    Parameters:
    -----------
    epoch : int
        The current epoch number.
    batch_idx : int
        The index of the current batch in the current epoch.
    """
    if epoch < warmup_epochs:
        effective_epoch = epoch + float(batch_idx + 1) / len(train_dataloader)
        lr_adj = (effective_epoch * (hvd.size() - 1) / warmup_epochs + 1)
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


def get_checkpoint_info():
    """
    Determines the starting epoch number for resuming training.

    This function looks at the checkpoint files in the specified model directory
    and extracts the epoch number from filenames in the format `checkpoint-{epoch}.pth`.
    It returns the highest epoch number found, or 0 if no checkpoints are found. 
    In a distributed training setup with Horovod, it ensures that the starting epoch
    number is broadcasted across all workers.

    Parameters:
    -----------
    args : argparse.Namespace
        The argument namespace containing the `pytorch_model_dir` (path to model directory).

    Returns:
    --------
    int
        The starting epoch number for training.
    """
    checkpoint_found = False
    start_checkpoint_num = 0
    epochs_already_trained_list = []
    filenames = os.listdir(args.pytorch_model_dir)
    
    # Check for existing checkpoint files and extract epoch numbers
    for file in filenames:
        model_file_name = re.findall(r"checkpoint\-(\d+)\.pth", file)
        # making sure list is not [] and
        if model_file_name:
            checkpoint_found = True
            epochs_already_trained_list.append(int(model_file_name[0]))
    
    # if no checkpoint is found return start_checkpoint_num = 0
    if not checkpoint_found:
        return checkpoint_found, start_checkpoint_num
    
    # Determine the starting epoch number
    start_checkpoint_num = max(epochs_already_trained_list) if epochs_already_trained_list else 0    

    return checkpoint_found, start_checkpoint_num

        
        
class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label):
        """
        Initializes the CelebAMaskHQ dataset.

        Parameters:
        -----------
        img_path : str
            Path to the directory containing the images.
        label_path : str
            Path to the directory containing the labels.
        transform_img : callable
            A function to apply transformations to the images.
        transform_label : callable
            A function to apply transformations to the labels.
        """
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.dataset = []
        self.preprocess()
        
        self.num_images = len(self.dataset)
    def preprocess(self):
        """
        Preprocesses the images and labels by collecting their paths 
        and pairing each image with its corresponding label.

        This function stores the paths of images and labels in the dataset 
        list, with each entry containing a tuple of the image path and 
        corresponding label path.
        """

        images = os.listdir(self.img_path)
        for img in images:
            img_path = os.path.join(self.img_path,img)
            label_path = os.path.join(self.label_path,img[:-4]+".png")
            self.dataset.append([img_path,label_path])
        return self.dataset
    
    def __getitem__(self,idx):
        """
        Retrieves the image and label at the given index, applies the 
        specified transformations, and returns them.

        Parameters:
        -----------
        idx : int
            The index of the image-label pair to retrieve.

        Returns:
        --------
        list : 
            A list containing the transformed image and transformed label.
        """
        img_path, label_path = self.dataset[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        return [self.transform_img(img),self.transform_label(label)]
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
        --------
        int : 
            The number of images in the dataset.
        """

        return self.num_images

def setup_data_loaders(transform_img, transform_label):
    """
    Sets up the training and testing data loaders with distributed training support.

    This function initializes the CelebAMaskHQ dataset, splits it into training and 
    testing datasets based on the specified percentage, and creates distributed samplers
    for both datasets. It then returns the training and testing data loaders.

    Parameters:
    -----------
    args : argparse.Namespace
        The argument namespace containing paths for the dataset, labels, and other training configurations.
    transform_img : callable
        The transformation to apply to the input images.
    transform_label : callable
        The transformation to apply to the label images.

    Returns:
    --------
    train_dataloader : torch.utils.data.DataLoader
        The data loader for the training set.
    test_dataloader : torch.utils.data.DataLoader
        The data loader for the testing set.
    """
    
    # Initialize the dataset
    dataset = CelebAMaskHQ(
        img_path=args.dataset_dir,
        label_path=args.label_dir,
        transform_img=transform_img,
        transform_label=transform_label
    )
    
    # Calculate dataset lengths for splitting
    dataset_len = len(dataset)
    test_data_len = int(dataset_len * args.test_split_perc)
    
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [dataset_len - test_data_len, test_data_len])
    
    # Create distributed samplers for data parallel training
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    # Create data loaders for both training and testing sets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)
    
    return train_dataloader, train_sampler, test_dataloader
    
            
def initialize_and_modify_model():
    """
    Initializes the FCN ResNet50 model and modifies the classifier layer.

    The model is initialized with pretrained weights only on the root rank (rank 0) in a 
    distributed training setup using Horovod. The classifier is replaced with a custom 
    one to adapt to the specific task (e.g., segmentation with 19 classes).

    Parameters:
    -----------
    args : argparse.Namespace
        The argument namespace containing any necessary configuration (e.g., device setup).

    Returns:
    --------
    model : torch.nn.Module
        The initialized and modified FCN ResNet50 model.
    """
    # Initialize the model without pretrained weights
    model = fcn_resnet50(pretrained=False)

    # Only the root process (rank 0) will download the pretrained weights
    if hvd.rank() == 0:
        model = fcn_resnet50(pretrained=True)

    # Modify the classifier module to match the task-specific requirements
    classifier = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Dropout(p=0.1, inplace=False),
        nn.Conv2d(512, 19, kernel_size=(1, 1), stride=(1, 1))  # Assuming 19 classes for output
    )
    
    # Replace the classifier module of the FCN model with the new one
    model.classifier = classifier

    return model


def initialize_optimizer(model):
    """
    Initializes the optimizer for the given model and applies Horovod's distributed optimizer.

    The function creates a Stochastic Gradient Descent (SGD) optimizer with momentum for 
    training the model. It then wraps the optimizer with Horovod's DistributedOptimizer 
    to handle gradient averaging across multiple processes.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to be optimized.

    Returns:
    --------
    optimizer : torch.optim.Optimizer
        The initialized and distributed optimizer.
    """
    # Initialize the SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)

    # Wrap the optimizer with Horovod's DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    return optimizer


def load_checkpoint(model, optimizer, checkpoint_found, start_checkpoint_num=0):
    """
    Loads the model and optimizer state from a checkpoint and broadcasts the state across all workers.

    If a checkpoint is found, it loads the model's state dictionary and optimizer state dictionary 
    from the specified checkpoint file. Then, it broadcasts the model parameters and optimizer state 
    to all workers using Horovod.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to be loaded from the checkpoint.
    optimizer : torch.optim.Optimizer
        The optimizer whose state is to be loaded from the checkpoint.
    args : argparse.Namespace
        The argument namespace containing checkpoint file paths and directories.
    start_checkpoint_num : int, optional (default=0)
        The epoch number to load the checkpoint for. If no checkpoint is found, the epoch is set to this value.

    Returns:
    --------
    model : torch.nn.Module
        The model with the loaded state.
    optimizer : torch.optim.Optimizer
        The optimizer with the loaded state.
    """
    checkpoint_found = False

    # Check for existing checkpoint files and load if available
    filepath = args.pytorch_checkpoint_format.format(epoch=start_checkpoint_num)
    filepath = args.pytorch_model_dir + filepath
    if checkpoint_found and hvd.rank() == 0:
        print("LOADING MODEL :- " + filepath)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("LOADED MODEL :- " + filepath)

    return model, optimizer




if __name__ == '__main__':
    args = parser.parse_args()
    # create directory to save pytorch model if folder is not already present
    makeDir(args.pytorch_model_dir)
    # create directory to save onnx model if folder is not already present
    makeDir(args.onnx_model_dir)
    # learning rate warmup
    warmup_epochs = 5
    
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #Initiazie Horovod
    hvd.init()

    # Pin GPU to be used
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")

    # Transform functions to apply on images
    transform_img = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ])
    # Transform function to apply on label
    transform_label = transforms.Compose([transforms.ToTensor()])
    
    train_dataloader, train_sampler, test_dataloader = setup_data_loaders(transform_img, transform_label)


    # Initialize model
    model = initialize_and_modify_model()
    model.cuda()

    # Initialize optimizer
    optimizer = initialize_optimizer(model)


    start_checkpoint_num = 0
    print(f"args.resume = {args.resume}")
    # as only root rank will save the model file
    if args.resume and hvd.rank() == 0:
        checkpoint_found, start_checkpoint_num = get_checkpoint_info()
        if checkpoint_found:
            model, optimizer = load_checkpoint(model, optimizer, start_checkpoint_num)
            start_checkpoint_num += 1

    # Broadcast the model parameters and optimizer state to all workers
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # each epoch training and testing loop
    for epoch in range(start_checkpoint_num, args.epochs):
        train(epoch, model, train_sampler, train_dataloader, optimizer)
        test_loss, test_accuracy = test(model, test_dataloader)
        save_checkpoint(epoch, model, optimizer)
        save_onnx(epoch, model)

# command to run
# horovodrun -np 1 python horovod_train.py --dataset-dir /media/mnehete32/New\ Volume/college_intro_ai/dataset/CelebAMask-HQ/CelebA-HQ-img/ --label-dir /media/mnehete32/New\ Volume/college_intro_ai/dataset/CelebAMask-HQ/CelebAMaskHQ-mask/ --resume