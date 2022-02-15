import os
import numpy as np
import torch
from torch.autograd import Variable
from data_utils import grayscale_img_load, listdir

def testMVImageClassifier(dataset_path, model, info, pooling = 'mean', cuda=False, verbose=False):

    # save pytorch model to eval mode
    model.eval()
    if (cuda):
        model.cuda()
    
    test_err = 0
    count = 0
    print("=>Testing...")

    # for each category
    for idx, c in enumerate(info['category_names']):
        category_full_dir = os.path.join(dataset_path,c)
        shape_dirs        = listdir(category_full_dir)
        print('=>Loading shape data: %s'%(c))

        # for each shape
        for s in shape_dirs:
            if verbose: print('=>Loading shape data: %s %s'%(s, c))
            views = listdir(os.path.join(category_full_dir, s))
            scores = np.zeros((len(views),len(info['category_names'])))
            count  += 1

            # for each view
            for i, v in enumerate(views):
                image_full_filename = os.path.join(category_full_dir, s, v)
                if 'png' not in image_full_filename : continue
                if verbose: print(' => Loading image: %s ...'%image_full_filename)
                im  = grayscale_img_load(image_full_filename)/255.
                im -= info['data_mean']
                im  = Variable(torch.from_numpy(im.astype('float32')), requires_grad=False).unsqueeze(0)
                # get predicted scores for each view
                if (cuda):
                    im = im.cuda()
                    scores[i, :] = model(im).detach().cpu().numpy().squeeze()
                else:
                    scores[i, :] = model(im).detach().numpy().squeeze()

            ''' 
            YOUR CODE GOES HERE
            1) Get category predictions per shape and test error averaged over all the test shapes.
            2) Implement 2 strategies: 1) mean and 2) max view-pooling by specifying input arg 'pooling', like
               >> pooling = 'mean' or pooling = 'max'
            
             '''

            if pooling == 'mean':
                avgs = np.average(scores, axis=0)
                predicted_label = np.argmax(avgs, axis=0)
            elif pooling == 'max':
                maxp = np.amax(scores, axis=0)
                predicted_label = np.argmax(maxp, axis=0)
            else:
                print('Unsupported prediction strategy!')
                return 

            if predicted_label != idx:
                test_err += 1

            if verbose: print('predicted label:  %s, ground-truth label: %s\n'%(info['category_names'][predicted_label] ,c))

    test_err = test_err / count
    print('Test error: %f%%\n'%(test_err * 100))