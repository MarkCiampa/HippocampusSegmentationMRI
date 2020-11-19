from scipy.stats import wilcoxon
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt



def wilcoxon_test(best_loss, best_net):
    base_file_path = "logs"
    input= os.path.join(base_file_path,best_net, best_loss) # logs/vnet/MDLoss
    net = ['unet','vnet']
    loss = ['CELoss','WCELoss','WCELoss(somma)', 'WCELoss(log)','MDLoss','MTLoss(0.4, 0.6)','MTLoss(0.3, 0.7)','MTLoss(0.2, 0.8)','MTLoss(0.1, 0.9)']
    p_list1 =  list()
    p_list2 = list()
    p_array_loss = []
    p_array_net = []


    input_path = os.path.join(base_file_path, best_net) 
    best_model_path = os.path.join(input_path, best_loss) # path di input

    input0 = best_model_path + "/dices.npy"
    input1 = best_model_path + "/anterior.npy"
    input2 = best_model_path + "/posterior.npy"
    input_np0 = np.load(input0)
    input_np1 = np.load(input1)
    input_np2 = np.load(input2)


    for i in range(2):
        
        net_ = net[i]
        half_file_path = os.path.join(base_file_path, net_)
        
        for loss_ in loss:
            #loss_ = loss[j]
            file_path = os.path.join(half_file_path, loss_) 
            if(file_path != best_model_path):

                    
                    file0 = file_path + '/dices.npy' 
                    file1 = file_path + '/anterior.npy'
                    file2 = file_path + '/posterior.npy'
                    file_np0 = np.load(file0)
                    file_np1 = np.load(file1)
                    file_np2 = np.load(file2)
                    
                    print("Test for: " + net_)
                    
                    w, p = wilcoxon(input_np0, file_np0, alternative='greater', zero_method='pratt')
                    print('Test background for loss : '+ loss_ + " p-value = {:.4f}".format(p))
                    p_array_loss.append(p)
                    
                    w, p = wilcoxon(input_np1, file_np1, alternative='greater', zero_method='pratt')
                    print('Test anterior for loss :'+ loss_ + " p-value = {:.4f}".format(p))
                    p_array_loss.append(p)
                    
                    w, p = wilcoxon(input_np2, file_np2, alternative='greater', zero_method='pratt')
                    print('Test posterior for loss :'+ loss_ + " p-value = {:.4f}".format(p))
                    p_array_loss.append(p)

                    print()
                    print()
                    

                    

                
                    p_list2.append(p_array_loss)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run Training on Hippocampus Segmentation")
    parser.add_argument(
        "-l",
        "--best_loss",
        default="",
        help="Insert the model that you want test!"
    )
    parser.add_argument(
        "-n",
        "--best_net",
        default="",
        help="Insert the model that you want test!"
    )
  

    args = parser.parse_args()
    best_loss = args.best_loss
    best_net = args.best_net
    
    wilcoxon_test(best_loss, best_net)

    






