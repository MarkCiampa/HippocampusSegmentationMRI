import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semseg.loss import get_multi_dice_loss
from config.config import LEARNING_RATE_REDUCTION_FACTOR
from semseg.utils import multi_dice_coeff
from config.config import *







def train_model(net, optimizer, train_data, config, model_path, weights=None, device=None, logs_folder=None, is_complete=False):
    name = config.loss
    print('Start training...')
    net = net.to(device)
    # train loop
    for epoch in range(config.epochs):

        epoch_start_time = time.time()
        running_loss = 0.0

        # lower learning rate
        if epoch == config.low_lr_epoch:
            for param_group in optimizer.param_groups:
                config.lr = config.lr / LEARNING_RATE_REDUCTION_FACTOR
                param_group['lr'] = config.lr

        # switch to train mode
        net.train()

        for i, data in enumerate(train_data):

            inputs, labels = data['t1']['data'], data['label']['data']
            if config.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # forward pass
            outputs = net(inputs)
            if name == 'MDLoss':
                outputs = F.softmax(outputs, dim=1)
            elif name == 'MTLoss':
                outputs = F.softmax(outputs, dim=1)
            
            

            if name == 'MDLoss':
                loss = get_multi_loss(outputs, labels, loss=name, device=device)
        
            elif name == 'CELoss':
                loss_function = nn.CrossEntropyLoss()
                labels = labels[:,0]
                loss = loss_function(outputs, labels.long())
                
            elif name == 'WCELoss':
                loss_function = nn.CrossEntropyLoss(weights)
                labels = labels[:,0]
                loss = loss_function(outputs, labels.long())
                
            elif name == 'MTLoss':
                loss = get_multi_loss(outputs, labels, loss=name, device=device)
                

            # empty gradients, perform backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save and print statistics
            
            running_loss += loss.data
            
            
            
        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time

        # print statistics
        print('  [Epoch {:04d}] - Train loss: {:.4f} - Time: {:.1f}'.format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))
        
       
    


        # switch to eval mode
        net.eval()

        # only validate every 'val_epochs' epochs
        if epoch % config.val_epochs == 0:
            if logs_folder is not None:
                checkpoint_path = os.path.join(model_path, 'model_epoch_{:04d}.pht'.format(epoch))
                torch.save(net.state_dict(), checkpoint_path)

    print('Training ended!')
    return net
    


def val_model(net, val_data, config, device=None):

    print("Start Validation...")
    net = net.to(device)
    # val loop
    multi_dices = list()
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(val_data):
            print("Iter {} on {}".format(i+1,len(val_data)))

            inputs, labels = data['t1']['data'], data['label']['data']
            if config.cuda: inputs, labels = inputs.cuda(), labels.cuda()

            # forward pass
            outputs = net(inputs)
            outputs = torch.argmax(outputs, dim=1)  #     B x Z x Y x X
            outputs_np = outputs.data.cpu().numpy() #     B x Z x Y x X
            labels_np = labels.data.cpu().numpy()   # B x 1 x Z x Y x X
            labels_np = labels_np[:,0]              #     B x Z x Y x X

            multi_dice = multi_dice_coeff(labels_np,outputs_np,config.num_outs)
            multi_dices.append(multi_dice)
    multi_dices_np = np.array(multi_dices)
    mean_multi_dice = np.mean(multi_dices_np)
    std_multi_dice = np.std(multi_dices_np)
    
    print("Multi-Dice: {:.4f} +/- {:.4f}".format(mean_multi_dice,std_multi_dice))
    return multi_dices, mean_multi_dice, std_multi_dice
