import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import glob
import h5py #type: ignore
import scipy.interpolate as spi #type: ignore
import scipy.constants as sc #type: ignore
import os

path = '/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/'
abmod_files = glob.glob('/home/hakon/Documents/abmod/runs/run_ 1*.txt')

m_add = 0
files = glob.glob(path + "*.h5")
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_0_30.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_30_60.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_60_90.h5')


def compare_snr(za_idx, vel_idx, files, mass='500', uncertainty='', plot=True, combine_drop=False):
    zas = ['0_', '10', '30', '50', '70']
    vels = ['10', '20', '30', '40', '50', '60']
    v2s = ['20', '30', '40', '50', '60', '73']
    vel = vels[vel_idx]
    za = zas[za_idx]
    alt = np.arange(70, 131, 1)
    za_file =[tf[tf.find("za_")+3:tf.find("za_")+5]for tf in files]
    file = ''
    for i in range(len(files)):
        if za_file[i] == za:
            file = files[i]
            break
    if file == '':
        print(f'No file found for ZA: {za}')
        return
    with h5py.File(file, 'r') as hf:
        keys = list(hf.keys())
        vel_files=[]
        for key in keys:
            v = key[key.find('drop_')+5:key.find('drop_')+7]
            if v == vel:
                vel_files.append(key)
        if 'nodrop' in vel_files[0]:
            nodrop_key = vel_files[0]
            drop_key = vel_files[1]
        else:
            nodrop_key = vel_files[1]
            drop_key = vel_files[0]
        drop_snr = hf[drop_key][()]
        nodrop_snr = hf[nodrop_key][()]
    #abmod_files = glob.glob('/home/hakon/Documents/abmod/runs/run_ '+mass+'*.txt')
    abmod_zas = ['0','20','40','60','80']
    abmod_vs = ['15','25', '35','45','55','65']
    abmod_file = '/home/hakon/Documents/abmod/runs/run_ ' +mass +'_ '+ abmod_vs[vel_idx] +'_ ' + abmod_zas[za_idx] + '.txt'
    aalt, asnr = np.loadtxt(abmod_file, unpack=True)
    aalt = aalt[::-1]
    asnr = asnr[::-1]
    f = spi.interp1d(aalt, asnr, kind='linear', fill_value='zero')
    abmod_snr = f(alt)

    if combine_drop:
        snr = np.concatenate((drop_snr, nodrop_snr), axis=0)
        mean_snr = np.mean(snr, axis=0)

        if uncertainty == 'STD':
            unc_snr = np.std(snr, axis=0)
        elif uncertainty == 'SE':
            unc_snr = np.std(snr, axis=0)/np.sqrt(len(snr))
        elif uncertainty == '95% CI':
            unc_snr = np.std(snr, axis=0)/np.sqrt(len(snr))*1.96
        elif uncertainty == '':
            unc_snr = np.zeros(len(alt))
        else:
            print('Uncertainty not recognized')
            unc_snr = np.zeros(len(alt))
        if not plot: #return dict with snr and uncertainty
            return {'snr': mean_snr, 'uncertainty': unc_snr, 'alt': alt, 'abmod_snr': abmod_snr}
        else:
            fig = plt.figure(figsize=(16,10))
            ax = fig.add_axes([0.05,0.05, 0.9, 0.9])
            ax.plot(10*np.log10(mean_snr), alt, label='Mean SNR', color='k', linestyle='-')
            ax.fill_betweenx(alt, 10*np.log10(mean_snr-unc_snr), 10*np.log10(mean_snr+unc_snr), color='k', alpha=0.2, label=uncertainty)
            ax.plot(abmod_snr, alt, label='Abmod SNR', color='r', linestyle='--')
            ax.set_xlabel('SNR [dB]')
            ax.set_ylabel('Altitude [km]')
            ax.set_title(f'ZA: {abmod_zas[za_idx]} Vel: {abmod_vs[vel_idx]} Mass: {mass} No meteors: {len(snr)}')
            ax.legend()
            ax.set_xlim(-50, 50)
            # if directory does not exist, create it
            if not os.path.exists(f'/home/hakon/Documents/abmod/figs/{mass}/combined_snr_plots/'):
                os.makedirs(f'/home/hakon/Documents/abmod/figs/{mass}/combined_snr_plots/')
            plt.savefig(f'/home/hakon/Documents/abmod/figs/{mass}/combined_snr_plots/combined_snr_{abmod_zas[za_idx]}_{abmod_vs[vel_idx]}_{mass}.png')
            plt.close()

    if not combine_drop:
        fig = plt.figure(figsize=(16,10))
        ax1 = fig.add_axes([0.05,0.1, 0.42, 0.8])
        ax2 = fig.add_axes([0.55-0.02,0.1,0.42,0.8])

        mean_drop = np.mean(drop_snr, axis=0)
        mean_nodrop = np.mean(nodrop_snr, axis=0)
        if uncertainty == 'STD':
            unc_drop = np.std(drop_snr, axis=0)
            unc_nodrop = np.std(nodrop_snr, axis=0)
        elif uncertainty == 'SE':
            unc_drop = np.std(drop_snr, axis=0)/np.sqrt(len(drop_snr))
            unc_nodrop = np.std(nodrop_snr, axis=0)/np.sqrt(len(nodrop_snr))
        elif uncertainty == '95% CI':
            unc_drop = np.std(drop_snr, axis=0)/np.sqrt(len(drop_snr))*1.96
            unc_nodrop = np.std(nodrop_snr, axis=0)/np.sqrt(len(nodrop_snr))*1.96
        elif uncertainty == '':
            unc_drop = np.zeros(len(alt))
            unc_nodrop = np.zeros(len(alt))
        else:
            print('Uncertainty not recognized')
            unc_drop = np.zeros(len(alt))
            unc_nodrop = np.zeros(len(alt))
        if plot:
            ax1.plot(10*np.log10(mean_drop), alt, label='Mean SNR', color='k', linestyle='-')
            ax1.fill_betweenx(alt, 10*np.log10(mean_drop-unc_drop), 10*np.log10(mean_drop+unc_drop), color='k', alpha=0.2, label=uncertainty)
            ax1.plot(abmod_snr, alt, label='Abmod SNR', color='r', linestyle='--')
            ax1.set_xlabel('SNR [dB]')
            ax1.set_ylabel('Altitude [km]')
            ax1.set_title(f'ZA: {abmod_zas[za_idx]} Vel: {abmod_vs[vel_idx]} Mass: {mass} $N_{{drop}}$ = {len(drop_snr)}')
            ax1.legend()
            ax1.set_xlim(-50, 50)
            ax2.plot(10*np.log10(mean_nodrop), alt, label='Mean SNR', color='k', linestyle='-')
            ax2.fill_betweenx(alt, 10*np.log10(mean_nodrop-unc_nodrop), 10*np.log10(mean_nodrop+unc_nodrop), color='k', alpha=0.2, label=uncertainty)
            ax2.plot(abmod_snr, alt, label='Abmod SNR', color='r', linestyle='--')
            ax2.set_xlabel('SNR [dB]')
            ax2.set_ylabel('Altitude [km]')
            ax2.set_title(f'ZA: {abmod_zas[za_idx]} Vel: {abmod_vs[vel_idx]} Mass: {mass} $N_{{nodrop}}$ = {len(nodrop_snr)}')
            ax2.legend()
            ax2.set_xlim(-50, 50)
            # if directory does not exist, create it
            if not os.path.exists(f'/home/hakon/Documents/abmod/figs/{mass}/snr_plots/'):
                os.makedirs(f'/home/hakon/Documents/abmod/figs/{mass}/snr_plots/')
            plt.savefig(f'/home/hakon/Documents/abmod/figs/{mass}/snr_plots/snr_{abmod_zas[za_idx]}_{abmod_vs[vel_idx]}_{mass}.png')
            plt.close()
        if not plot:
            return {'snr_drop': mean_drop, 'uncertainty_drop': unc_drop, 'snr_nodrop': mean_nodrop, 'uncertainty_nodrop': unc_nodrop, 'alt': alt, 'abmod_snr': abmod_snr}


for zi in range(5):
    for vi in range(6):
        compare_snr(zi, vi, files, mass='500', uncertainty='SE', plot=True, combine_drop=True)
        compare_snr(zi, vi, files, mass='500', uncertainty='SE', plot=True, combine_drop=False)
        compare_snr(zi, vi, files, mass='1', uncertainty='SE', plot=True, combine_drop=True)
        compare_snr(zi, vi, files, mass='1', uncertainty='SE', plot=True, combine_drop=False)

