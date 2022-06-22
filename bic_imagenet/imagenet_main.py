import torch
import numpy as np
from imagenet_trainer import Trainer
import sys
from utils import *
import argparse
import utils_imagenet
from matplotlib import image
#from exemplar import Exemplar
#import torchvision.transforms as transforms
#from dataset import BatchData
parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 256, type = int)
parser.add_argument('--epoch', default = 100, type = int)
parser.add_argument('--lr', default = 0.1, type = int)
#parser.add_argument('--max_size', default = 2000, type = int)
parser.add_argument('--total_cls', default = 100, type = int)
args = parser.parse_args()

def get_image_shape(img):
    shape = img
    return len(shape[0]), len(shape[1])


def resize_image_shape(img, output):
    width, height = get_image_shape(img)

    img=np.resize(img, (output, output, 3))

    return img

if __name__ == "__main__":

    max_size = 2000

    #If True : ImageNet 1000
    FLAG_1000 = False
    
    ImgSize = 224
    NumChannels = 3
    
    if FLAG_1000 == True:
        NumClasses = 1000
    else:
        NumClasses = 100

    NumTrainFiles =1024
    Buffer = 1500

    # Adjustable parameters
    NumVal = 0.1

    if FLAG_1000 == True:
        IncrementalStep = 100
    else:
        IncrementalStep = 10

    groups = 10
    NumProto = 200

    if FLAG_1000 == True:
        DataDir = '/workspace/junsu/LargeScaleIncrementalLearning/dataImageNet1000/'
    else:
        DataDir = '/workspace/junsu/LargeScaleIncrementalLearning/dataImageNet100/'

    np.random.seed(1993) # same with iCaRL

    #######################################

    assert (NumClasses == IncrementalStep * groups)

    print("Mixing the classes and putting them in batches of classes")

    order = np.arange(NumClasses)
    np.random.shuffle(order)

    xTrainProto=[]
    yTrainProto=[]

    for _ in range(NumClasses):
        xTrainProto.append([])
        yTrainProto.append([])


    print("Loading all data")
    fpath = os.path.join(DataDir, 'train.txt')
    xTrain, yTrain = utils_imagenet.load_data(fpath, order)
    fpath = os.path.join(DataDir, 'val.txt')
    xTest, yTest = utils_imagenet.load_data(fpath, order)

    print(len(xTrain), len(yTrain), len(xTest), len(yTest))

    _NUM_IMAGES = {
            'train' : len(xTrain),
            'validation' : len(xTest),
            }

    print("Creating a validation set and generating group...")
    max_val = int(NumVal * groups * IncrementalStep * NumProto / IncrementalStep)
    print("max_val :", max_val)

    xTrain, yTrain, xVal, yVal, xTest, yTest = utils_imagenet.prepare_validation(xTrain,yTrain,xTest,yTest,groups, IncrementalStep, max_val)

    print("Image preprocessing is done")

    print(len(xTrain), len(yTrain), len(xVal), len(yVal), len(xTest), len(yTest))
    
    totalImage = 0
    for i in range(IncrementalStep):
        totalImage= totalImage + len(yTrain[i])

    print(totalImage)
    print(len(yVal))
    train_groups = [[],[],[],[],[],[],[],[],[],[]] #10groups

    val_groups = [[],[],[],[],[],[],[],[],[],[]] #10groups

    test_groups = [[],[],[],[],[],[],[],[],[],[]] #10groups
    
    for i in range(len(xTrain)):
        for j in range(len(xTrain[i])):
            trainpath=os.path.join(DataDir, xTrain[i][j])
            xdata=image.imread(trainpath)
            xdata=resize_image_shape(xdata, 256)
            train_groups[i].append((xdata, yTrain[i][j]))
    
    print(type(xdata))
    print(len(train_groups[0]))
    print(type(train_groups[0][0]))
    
    for k in range(groups):
        for i in range(IncrementalStep*k, IncrementalStep*(k+1)):
            for j in range(len(xVal[i])):
                valpath=os.path.join(DataDir, xVal[i][j])
                xdata=image.imread(valpath)
                xdata=resize_image_shape(xdata, 256)
                val_groups[k].append((xdata, yVal[i][j]))
    
    print(len(val_groups[0]))
    
    for i in range(len(xTest)):
        for j in range(len(xTest[i])):
            testpath=os.path.join(DataDir, xTest[i][j])
            xdata=image.imread(testpath)
            xdata=resize_image_shape(xdata, 256)
            test_groups[i].append((xdata, yTest[i][j]))
    
    print(len(test_groups[0]))
    print("preprocessing is done")
    print("Start training...")
    trainer = Trainer(train_groups, val_groups, test_groups, args.total_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, max_size)

"""
    exemplar = Exemplar(max_size, args.total_cls)

    transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])




    for i in range(10):
        for j in range(10):
            trainpath = os.path.join(DataDir, xTrain[i][j])
            xdata=image.imread(trainpath)
            xdata=resize_image_shape(xdata, 224)
          
            print("xdata:", xdata.shape)

            
            
            train_groups[i].append((xdata,yTrain[i][j]))

    for i in range(10):
        print("group number:", i)
        train_local = train_groups[i]
        print(len(train_local))
        train_x, train_y = zip(*train_local)

        train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_ys.extend(train_y)
        
        print(len(train_xs[0]))
        
        eoc_train = DataLoader(BatchData(train_xs, train_ys, transforms), batch_size = 1, shuffle=False, drop_last=True)
        """
 
