import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rcParams.update({'font.family': 'serif',
                    'font.size': 12*2,
                    'axes.labelsize': 12*2,
                    'xtick.labelsize': 11*2,
                    'ytick.labelsize': 11*2,
                    'legend.fontsize': 12*2,
                    'figure.titlesize': 12*2
})
fig_width=6.733*2
ls_d = 'solid'
ls_nd = 'solid'
ls_mod = 'dashed'
c_d = 'red'
c_nd = 'k'
c_mod = 'blue'
error_alpha=0.2
lw= 2
"""path = '/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/USE_snr0_anem100_c0.5/'
files = glob.glob(path + '*.h5')


for f in files:
    with h5py.File(f,'r') as hf:
        rcs_drop_10_20 = hf['rcs_drop_10_20'][()]
        rcs_drop_20_30 = hf['rcs_drop_20_30'][()]
        rcs_drop_30_40 = hf['rcs_drop_30_40'][()]
        rcs_drop_40_50 = hf['rcs_drop_40_50'][()]
        rcs_drop_50_60 = hf['rcs_drop_50_60'][()]
        rcs_drop_60_73 = hf['rcs_drop_60_73'][()]
        rcs_nodrop_10_20 = hf['rcs_nodrop_10_20'][()]
        rcs_nodrop_20_30 = hf['rcs_nodrop_20_30'][()]
        rcs_nodrop_30_40 = hf['rcs_nodrop_30_40'][()]
        rcs_nodrop_40_50 = hf['rcs_nodrop_40_50'][()]
        rcs_nodrop_50_60 = hf['rcs_nodrop_50_60'][()]
        rcs_nodrop_60_73 = hf['rcs_nodrop_60_73'][()]
        tot_len = len(rcs_drop_10_20)
        tot_len += len(rcs_drop_20_30)
        tot_len += len(rcs_drop_30_40)
        tot_len += len(rcs_drop_40_50)
        tot_len += len(rcs_drop_50_60)
        tot_len += len(rcs_drop_60_73)
        tot_len += len(rcs_nodrop_10_20)
        tot_len += len(rcs_nodrop_20_30)
        tot_len += len(rcs_nodrop_30_40)
        tot_len += len(rcs_nodrop_40_50)
        tot_len += len(rcs_nodrop_50_60)
        tot_len += len(rcs_nodrop_60_73)
        print(tot_len)"""



file_149 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/0_149.txt', skiprows=1, usecols=(5,11))
file_299 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/149_299.txt', skiprows=1, usecols=(5,11))
file_449 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/300_449.txt', skiprows=1, usecols=(5,11))
file_600 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/450_600.txt', skiprows=1, usecols=(5,11))
file_750 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/600_749.txt', skiprows=1, usecols=(5,11))
file_899 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/750_899.txt', skiprows=1, usecols=(5,11))
file_1000 = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/msis_plot/900_1000.txt', skiprows=1, usecols=(5,11))

altitude = np.concatenate((file_149[:,0], file_299[:,0], file_449[:,0], file_600[:,0], file_750[:,0], file_899[:,0], file_1000[:,0]))
density = np.concatenate((file_149[:,1], file_299[:,1], file_449[:,1], file_600[:,1], file_750[:,1], file_899[:,1], file_1000[:,1]))

# convert density from g/cm^3 to kg/m^3
density = density * 1000

fig = plt.figure(figsize=(fig_width, 8))
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.111, right=0.972, top=0.966, bottom=0.11, wspace=0.2, hspace=0.2)
ax.plot(density, altitude, color='k', lw=lw, ls=ls_d, label='Atmospheric density')
ax.set_xlabel('Density (kg/m$^3$)', loc='right')
ax.set_ylabel('Altitude (km)', loc='top')
ax.set_ylim(0, 1000)
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=12*2, frameon=False)
ax.tick_params(direction='in')
plt.savefig('/home/hakon/Documents/abmod/msis.png', dpi=300)
plt.close()




test_acc = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/dl_plots/test_acc.txt', delimiter=',')
train_acc = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/dl_plots/train_acc.txt', delimiter=',')
test_loss = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/dl_plots/test_loss.txt', delimiter=',')
train_loss = np.loadtxt('/home/hakon/Documents/meteor_fork/hakon/master/dl_plots/train_loss.txt', delimiter=',')


"""with h5py.File('/home/hakon/Documents/meteor_fork/data/2020/01/kep_collect.h5', 'r') as hf:
    cnn_input = hf['cnn_input'][()]
ex = cnn_input[1]
plt.subplot(711)
plt.plot(ex[0],'k.')#rcs
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(712)
plt.plot(ex[1],'k.')#snr
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(713)
plt.plot(ex[2],'k.')#zpp
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(714)
plt.plot(ex[3],'k.')#beam pos
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(715)
plt.plot(ex[4],'k.')#pos_e
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(716)
plt.plot(ex[5],'k.')#pos_n
plt.xlim(0, 256)
plt.ylim(0, 1)
plt.subplot(717)
plt.plot(ex[6],'k.')#pos_u
plt.xlim(0, 256)
plt.ylim(0, 1)
# define the font and size for x and y axis labels
plt.show()"""






cm_test = np.array([[446,41],[104,390]])
cm_train = np.array([[1911,83],[113,1894]])
#cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
#cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(1, 2, figsize=(fig_width, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['0', '1'])
disp.plot(ax=ax[1], cmap=plt.cm.Blues, colorbar=False)
disp.ax_.set_title('Test set')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['0', '1'])
disp.plot(ax=ax[0], cmap=plt.cm.Blues, colorbar=False)
disp.ax_.set_title('Training set')
fig.subplots_adjust(left=0.062, right=0.969, top=0.954, bottom=0.11, wspace=0.325, hspace=0.2)
#print precision and recall
precision = cm_test[1, 1] / (cm_test[1, 1] + cm_test[0, 1])
recall = cm_test[1, 1] / (cm_test[1, 1] + cm_test[1, 0])
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
plt.savefig('/home/hakon/Documents/abmod/confusion_matrix.png', dpi=300)
plt.close()



print(max(test_acc))

fig = plt.figure(figsize=(fig_width, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(left=0.07, right=0.972, top=0.91, bottom=0.11, wspace=0.3, hspace=0.2)

ax1.plot(train_acc*100, label='Training set', c='k')
ax1.plot(test_acc*100, label='Test set', c='r')
ax1.set_xlabel('Epoch', loc='right')
ax1.set_ylabel('Accuracy (%)',loc='top')
ax1.legend(loc='upper left', bbox_to_anchor=(0,1.1), frameon=False,ncol=2,borderaxespad=0.0,handletextpad=0.3)
ax1.tick_params(direction='in')
ax1.set_xlim(0,12000)

train_loss = train_loss/(5002*0.8/256)
test_loss = test_loss/(5002*0.2/256)
ax2.plot(train_loss, label='Train Loss', c='k')
ax2.plot(test_loss, label='Test Loss', c='r')
ax2.set_xlabel('Epoch', loc='right')
ax2.set_ylabel('Loss',loc='top')
ax2.set_xlim(0,12000)
ax2.tick_params(direction='in')

plt.savefig('/home/hakon/Documents/abmod/dl_training.png', dpi=300)





