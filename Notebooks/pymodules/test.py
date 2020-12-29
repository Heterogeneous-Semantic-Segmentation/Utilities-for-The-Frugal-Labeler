import pickle
import numpy as np
import matplotlib.pyplot as plt
from .data import train_generator


for i in [0.0,0.4,0.7]:
    for j in [0.0,0.5,1.0]:
        file = open("alpha/Missing_%.1f_alpha_%.1f/history" %(i,j), 'rb')
        first = pickle.load(file)
        file.close()
        val_loss_1 = first.get('val_loss')
        acc_1 = first.get('accuracy')

        arr_losses = np.array([min(val_loss_1)])
        # print('Dice-Coeff: %.2f+-%.2f'%(1-np.mean(arr_losses),np.std(arr_losses)))
        accs = np.array([acc_1[np.argmin(val_loss_1)]])
        # print('Acc:        %.2f+-%.2f'%(np.mean(accs)*100,np.std(accs)*100))
        epochs = np.array([np.argmin(val_loss_1)])
        # print('Epoch: %d-%.2f'%(int(np.mean(epochs)),np.std(epochs)))

        # print('')

        dropped_ammount = i * 100
        argmin_epochs = int(np.mean(epochs))
        argmin_epochs_std = np.std(epochs)
        acc = np.mean(accs) * 100
        acc_std = np.std(accs) * 100
        dice = 1 - np.mean(arr_losses)
        dice_std = np.std(arr_losses)
        print('%d & $%2.d \pm %05.2f$  & $%.2f \pm %.2f$ & $%.4f \pm %.2f$ \\\\' % (dropped_ammount,
                                                                                    argmin_epochs,
                                                                                    argmin_epochs_std,
                                                                                    acc,
                                                                                    acc_std,
                                                                                    dice,
                                                                                    dice_std))





exit()

i = 0.0
while i<0.9:
    #print('----%1.f----'%(i*100))

    file = open("heteregenous_data_set_scenario/%.1f/history"%i,'rb')
    first = pickle.load(file)
    file.close()
    val_loss_1 = first.get('val_loss')
    acc_1 = first.get('accuracy')

    file = open("heteregenous_data_set_scenario_2/%.1f/history"%i,'rb')
    first = pickle.load(file)
    file.close()
    val_loss_2 = first.get('val_loss')
    acc_2 = first.get('accuracy')



    arr_losses = np.array([min(val_loss_1),min(val_loss_2)])
    #print('Dice-Coeff: %.2f+-%.2f'%(1-np.mean(arr_losses),np.std(arr_losses)))
    accs = np.array([acc_1[np.argmin(val_loss_1)],acc_2[np.argmin(val_loss_2)]])
    #print('Acc:        %.2f+-%.2f'%(np.mean(accs)*100,np.std(accs)*100))
    epochs = np.array([np.argmin(val_loss_1),np.argmin(val_loss_2)])
    #print('Epoch: %d-%.2f'%(int(np.mean(epochs)),np.std(epochs)))

    #print('')

    dropped_ammount = i*100
    argmin_epochs = int(np.mean(epochs))
    argmin_epochs_std = np.std(epochs)
    acc = np.mean(accs)*100
    acc_std = np.std(accs)*100
    dice = 1-np.mean(arr_losses)
    dice_std = np.std(arr_losses)
    print('%d & $%2.d \pm %05.2f$  & $%.2f \pm %.2f$ & $%.4f \pm %.2f$ \\\\'%(dropped_ammount,
                                                                     argmin_epochs,
                                                                     argmin_epochs_std,
                                                                    acc,
                                                                    acc_std,
                                                                    dice,
                                                                    dice_std))
    i+=0.1