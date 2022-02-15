from trainMVShapeClassifier import trainMVShapeClassifier
from testMVImageClassifier import testMVImageClassifier
import pickle as p
import torch

train_path = './dataset/train'
test_path = './dataset/test'

use_cuda = True 
verbose_flag = False 

# TRAIN
model, info = trainMVShapeClassifier(train_path, cuda=use_cuda, verbose=verbose_flag)
#
# # TO SAVE TIME for just testing code, uncomment the following 2 lines to load your pre-trained model
# model = torch.load('model/model_epoch_19.pth', map_location=lambda storage, location: storage)["model"]
# info = p.load( open( "info.p", "rb" ) )

# TEST
testMVImageClassifier(test_path, model, info, pooling='mean', cuda=use_cuda, verbose=verbose_flag)
testMVImageClassifier(test_path, model, info, pooling='max', cuda=use_cuda, verbose=verbose_flag)