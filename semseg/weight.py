import torch
import torch.nn as nn
from config.config import SemSegMRIConfig
import numpy as np
from torchio import Image, ImagesDataset, SubjectsDataset
import torchio
from config.augm import train_transform
from tabulate import tabulate



def TorchIODataLoader_train(image_val, label_val):
    #print('Building TorchIO Training Set Loader...')
    subject_list = list()
    for idx, (image_path, label_path) in enumerate(zip(image_val, label_val)):
        s1 = torchio.Subject(
            t1=Image(type=torchio.INTENSITY, path=image_path),
            label=Image(type=torchio.LABEL, path=label_path),
        )
        subject_list.append(s1)

    subjects_dataset = SubjectsDataset(subject_list, transform=train_transform)
    train_data = torch.utils.data.DataLoader(subjects_dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    #print('TorchIO Training Loader built!')
    return train_data

def calc_weight(kf):
    weights=list()

    for fold, (train_index, val_index) in enumerate(kf.split(SemSegMRIConfig.train_images)):

        train_images_np, train_labels_np = np.array(SemSegMRIConfig.train_images), np.array(SemSegMRIConfig.train_labels)
        train_images_list = list(train_images_np[train_index])
        train_labels_list = list(train_labels_np[train_index])
        train_data = TorchIODataLoader_train(train_images_list, train_labels_list)

        x0=0
        x1=0
        x2=0

        for i, data in enumerate(train_data):
            labels = data['label']['data']
            labels= labels.data.cpu().numpy()
            labels= labels [0,:,:,:]
            labels = labels.flatten()
            x0+=(labels==0).sum()
            x1+=(labels==1).sum()
            x2+=(labels==2).sum()
            xtot= (x0+x1+x2)
           
        frequency0= x0/xtot
        in_frequency0=xtot/x0

        frequency1= x1/xtot
        in_frequency1=xtot/x1

        frequency2= x2/xtot
        in_frequency2=xtot/x2

       # somma = in_frequency0 + in_frequency1 + in_frequency2
      #  weight = [(in_frequency0)/somma, (in_frequency1)/somma, (in_frequency2)/somma, xtot]
        weight = [np.log(in_frequency0), np.log(in_frequency1), np.log(in_frequency2), xtot]
        weights.append(weight)

       # wheights=torch.tensor( wheights)
        #wheights=wheights.float()



    train_data = TorchIODataLoader_train(SemSegMRIConfig.train_images, SemSegMRIConfig.train_labels)

    x0=0
    x1=0
    x2=0
    for i, data in enumerate(train_data):
        labels = data['label']['data']
        labels= labels.data.cpu().numpy()
        labels= labels [0,:,:,:]
        labels = labels.flatten()
        x0+=(labels==0).sum()
        x1+=(labels==1).sum()
        x2+=(labels==2).sum()
        xtot= (x0+x1+x2) 

    frequency0= x0/xtot
    in_frequency0=xtot/x0

    frequency1= x1/xtot
    in_frequency1=xtot/x1

    frequency2= x2/xtot
    in_frequency2=xtot/x2

    #somma = in_frequency0 + in_frequency1 + in_frequency2
   # weight = [(in_frequency0)/somma, (in_frequency1)/somma, (in_frequency2)/somma, xtot]
    weight = [np.log(in_frequency0), np.log(in_frequency1), np.log(in_frequency2), xtot]

    weights.append(weight)

    stamp_weights(weights)
    for i in range(len(weights)):
        weights[i].pop()
       
    return weights

def stamp_weights(weights):

    final_matrix=list()

    for i in range(len(weights)):
        if(i==len(weights)-1):
            intermediate_matrix=['TOTAL FOLD  ',weights[i][0], weights[i][1],weights[i][2],weights[i][3]]
        else:
            intermediate_matrix=['FOLD  '+i.__str__(),weights[i][0], weights[i][1],weights[i][2],weights[i][3]]
        final_matrix.append(intermediate_matrix)
        
    print(tabulate(final_matrix,headers=['FOLD', 'WEIGHT_0','WEIGHT_1','WEIGHT_2','N_VOXEL'])) 

