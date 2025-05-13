import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import glob
import h5py #type: ignore
import scipy.interpolate as spi #type: ignore
import scipy.constants as sc #type: ignore
import os
from tqdm import tqdm #type: ignore
import matplotlib.lines as mlines
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
abmod_line = mlines.Line2D([],[],color=c_mod, linestyle=ls_mod,markersize=8,label='Model')
nodrop_line = mlines.Line2D([],[],color=c_nd, linestyle=ls_d,markersize=8,label='No SNR drop')
drop_line = mlines.Line2D([],[],color=c_d, linestyle=ls_nd,markersize=8,label='SNR drop')
abmod_error = mlines.Line2D([],[],color=c_mod, linestyle='solid',markersize=8,linewidth=13, label='95% CI',alpha=error_alpha)
nodrop_error = mlines.Line2D([],[],color=c_nd, linestyle='solid',markersize=8,linewidth=13, label='95% CI',alpha=error_alpha)
drop_error = mlines.Line2D([],[],color=c_d, linestyle='solid',markersize=8,linewidth=13, label='95% CI',alpha=error_alpha)



path = '/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/'
abmod_files = glob.glob('/home/hakon/Documents/abmod/runs/run_ 1*.txt')

m_add = 0
files = glob.glob(path + "*.h5")
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_0_30.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_30_60.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_60_90.h5')


def compare_snr(za_idx, vel_idx, files, mass='500', uncertainty='', plot=True, combine_drop=False, compare_drops=False):
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

    if not combine_drop and not compare_drops:
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
            plt.close()
            return {'snr_drop': drop_snr, 'uncertainty_drop': unc_drop,
                    'snr_nodrop': nodrop_snr, 'uncertainty_nodrop': unc_nodrop, 
                    'alt': alt, 'abmod_snr': abmod_snr, 'mean_drop': mean_drop, 'mean_nodrop': mean_nodrop}
    
    if compare_drops:
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_axes([0.05,0.1, 0.9, 0.8])
        mean_drop = np.mean(drop_snr, axis=0)
        mean_nodrop = np.mean(nodrop_snr, axis=0)

        ax.plot(10*np.log10(mean_drop), alt, label='Drop', color='blue', linestyle='-')
        ax.plot(10*np.log10(mean_nodrop), alt, label='No Drop', color='k', linestyle='-')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Altitude [km]')
        ax.set_title(f'ZA: {abmod_zas[za_idx]} Vel: {abmod_vs[vel_idx]} Mass: {mass} $N_{{drop}}$ = {len(drop_snr)} $N_{{nodrop}}$ = {len(nodrop_snr)}')
        ax.legend()
        ax.set_xlim(-50, 50)
        # if directory does not exist, create it
        if not os.path.exists(f'/home/hakon/Documents/abmod/figs/drop_nodrop_comparison/'):
            os.makedirs(f'/home/hakon/Documents/abmod/figs/drop_nodrop_comparison/')
        plt.savefig(f'/home/hakon/Documents/abmod/figs/drop_nodrop_comparison/snr_{abmod_zas[za_idx]}_{abmod_vs[vel_idx]}.png')
        plt.close()


m001=[".01", ".02", ".03", ".04", ".05", ".06", ".07", ".08", ".09"]
m01=[".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
m1=["1", "2", "3", "4", "5", "6", "7", "8", "9"]
m10=["10", "20", "30", "40", "50", "60", "70", "80", "90"]
m100=["100", "200", "300", "400", "500", "600", "700", "800", "900"]
masses = m001 + m01 + m1 + m10 + m100
if False:
    for zi in range(5):
        for vi in range(6):
            try:
                compare_snr(zi, vi, files, compare_drops=True)
            except Exception as e:
                print(e)
                continue
            for m in masses:
                try:
                    compare_snr(zi, vi, files, mass=m, uncertainty='SE', plot=True, combine_drop=True)
                    compare_snr(zi, vi, files, mass=m, uncertainty='SE', plot=True, combine_drop=False)
                except Exception as e:
                    print(f'Error in mass {m} for ZA: {zi} Vel: {vi}')
                    print(e)
                    continue

#{'snr_drop': mean_drop, 'uncertainty_drop': unc_drop, 'snr_nodrop': mean_nodrop,
#'uncertainty_nodrop': unc_nodrop, 'alt': alt, 'abmod_snr': abmod_snr}

all_drop = np.zeros((1,61))
all_nodrop = np.zeros((1,61))
if False:
    for zi in range(5):
        for vi in range(6):
            #zi=3
            #vi=5
            try:
                snr_dict = compare_snr(zi, vi, files,plot=False, combine_drop=False)
                drop = snr_dict['snr_drop']
                nodrop = snr_dict['snr_nodrop']
                all_drop = np.concatenate((all_drop, drop), axis=0)
                all_nodrop = np.concatenate((all_nodrop, nodrop), axis=0)
                alt = snr_dict['alt']
            except Exception as e:
                pass
    
    all_drop = all_drop[1:,:]
    all_nodrop = all_nodrop[1:,:]
    # remove outlier by zeroing out the top 3 values at each altitude
    for i in range(61):
        for j in range(3):
            max_arg = np.argmax(all_drop[:,i])
            all_drop[max_arg,i] = 0
            max_arg = np.argmax(all_nodrop[:,i])
            all_nodrop[max_arg,i] = 0
    print(all_drop.shape)
    print(all_nodrop.shape)
    mean_drop = np.mean(all_drop, axis=0)
    mean_nodrop = np.mean(all_nodrop, axis=0)
    se_drop = 1.96*np.std(all_drop, axis=0)/np.sqrt(len(all_drop))
    se_nodrop = 1.96*np.std(all_nodrop, axis=0)/np.sqrt(len(all_drop))
    fig = plt.figure(figsize=(fig_width,10))
    ax = fig.add_axes([0.1,0.1,0.84,0.7])
    ax.plot(10*np.log10(mean_drop), alt, label='Drop', color=c_d, linestyle=ls_d)
    ax.fill_betweenx(alt, 10*np.log10(mean_drop-se_drop), 10*np.log10(mean_drop+se_drop), color=c_d, ls=ls_d, alpha=error_alpha, label='95% CI')
    ax.plot(10*np.log10(mean_nodrop), alt, label='No Drop', color='k', linestyle='-')
    ax.fill_betweenx(alt, 10*np.log10(mean_nodrop-se_nodrop), 10*np.log10(mean_nodrop+se_nodrop), color=c_nd, ls=ls_nd,alpha=error_alpha, label='95% CI')
    ax.set_xlabel('SNR (dB)',loc='right')
    ax.set_ylabel('Altitude (km)',loc='top')
    ax.legend(handles=[nodrop_line, nodrop_error,drop_line,drop_error], loc='upper left', bbox_to_anchor=(0,1.15), frameon=False,ncol=2,borderaxespad=0.0,handletextpad=0.3)
    ax.set_xlim(-50, 50)
    #plt.savefig('/home/hakon/Documents/abmod/new_imgs/mean_profile_d_nd',dpi=300)
    #plt.close()
    plt.show()

    all_meteors = np.concatenate((all_drop, all_nodrop), axis=0)
    mean_all = np.mean(all_meteors, axis=0)
    se_all = 1.96*np.std(all_meteors, axis=0)/np.sqrt(len(all_meteors))
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_axes([0.05,0.1, 0.9, 0.8])
    ax.plot(10*np.log10(mean_all), alt, label='All meteors', color='k', linestyle='-')
    ax.fill_betweenx(alt, 10*np.log10(mean_all-se_all), 10*np.log10(mean_all+se_all), color='k', alpha=0.2, label='SE')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'All meteors')
    ax.legend()
    ax.set_xlim(-50, 50)
    plt.show()





    """fig = plt.figure(figsize=(16,10))
    ax = fig.add_axes([0.05,0.1, 0.9, 0.8])
    for i in tqdm(range(len(all_nodrop))):
        ax.plot(10*np.log10(all_nodrop[i]), alt, color='k', alpha=0.1)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'All meteors')
    ax.set_xlim(0, 100)
    plt.show()
    plt.close()"""

    #create a 2d histogram with snr on x axis and altitude on y axis
    # need to duplicate alt len(all_nodrop) times
    alt_2d = np.zeros((len(all_nodrop), len(alt)))
    for i in range(len(all_nodrop)):
        alt_2d[i] = alt

    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_axes([0.05,0.1, 0.42, 0.8])
    ax2 = fig.add_axes([0.55-0.02,0.1,0.42,0.8])

    snr = 10*np.log10(all_nodrop.flatten())
    # set -inf to nan
    snr[np.isneginf(snr)] = np.nan
    # remove nan values
    alt_2d = alt_2d.flatten()
    alt_2d = alt_2d[~np.isnan(snr)]
    snr = snr[~np.isnan(snr)]
    
    y_bins = len(np.unique(alt_2d))
    h = ax2.hist2d(snr, alt_2d, bins=[100, y_bins], norm='log', cmap='viridis')
    ax2.set_xlabel('SNR [dB]')
    ax2.set_ylabel('Altitude [km]')
    ax2.set_title(f'Nodrop')
    ax2.set_xlim(15, 77)
    ax2.set_ylim(88, 129)
    cbar = fig.colorbar(h[3], ax=ax2)
    cbar.set_label('Counts')

    
    snr = 10*np.log10(all_drop.flatten())
    # set -inf to nan
    snr[np.isneginf(snr)] = np.nan
    # remove nan values
    alt_2d = np.zeros((len(all_drop), len(alt)))
    for i in range(len(all_drop)):
        alt_2d[i] = alt
    alt_2d = alt_2d.flatten()
    alt_2d = alt_2d[~np.isnan(snr)]
    snr = snr[~np.isnan(snr)]
    y_bins = len(np.unique(alt_2d))
    h = ax1.hist2d(snr, alt_2d, bins=[100, y_bins], norm='log', cmap='viridis')
    ax1.set_xlabel('SNR [dB]')
    ax1.set_ylabel('Altitude [km]')
    ax1.set_title(f'Drop')
    ax1.set_xlim(15, 77)
    ax1.set_ylim(88, 129)
    # add colorbar
    cbar = fig.colorbar(h[3], ax=ax1)
    cbar.set_label('Counts')
    plt.show()
    

if True:
    vels = ['10', '20', '30', '40', '50', '60']
    v2s = ['20', '30', '40', '50', '60', '73']
    zas = ['0', '10', '30', '50', '70','90']
    for zi in range(5):
        fig = plt.figure(figsize=(fig_width,10))
        ax = fig.add_axes([0.1,0.1,0.84,0.7])
        handles = []
        #ax.plot(10*np.log10(mean_drop), alt, label='Drop', color=c_d, linestyle=ls_d)

        for vi in range(6):
            #zi=3
            #vi=5
            try:
                snr_dict = compare_snr(zi, vi, files,plot=False, combine_drop=False)
                drop = snr_dict['snr_drop']
                nodrop = snr_dict['snr_nodrop']
                alt = snr_dict['alt']
            except Exception as e:
                pass
    
            snr = np.concatenate((drop,nodrop),axis=0)
            # remove outlier by zeroing out the top 3 values at each altitude
            remove = 5
            for i in range(61):
                for j in range(remove):
                    max_arg = np.argmax(snr[:,i])
                    snr[max_arg,i] = 0
            n_snr = (len(snr)-remove)
            mean_snr = np.sum(snr,axis=0)/n_snr
            snr_std = np.std(snr,axis=0,ddof=remove)
            snr_ci =1.96*snr_std/np.sqrt(n_snr-remove)
            ax.plot(10*np.log10(mean_snr),alt,label=f'({vels[vi]},{v2s[vi]})',linestyle='solid',color=(1-vi/5,0,0))
            handles.append(mlines.Line2D([],[],color=(1-vi/5,0,0), linestyle='solid', markersize=8, label=f'v$\in$({vels[vi]},{v2s[vi]})'))
            #ax.fill_betweenx(alt, 10*np.log10(mean_snr-snr_ci), 10*np.log10(mean_snr+snr_ci), color=c_d, ls=ls_d, alpha=error_alpha, label='95% CI')
        ax.set_xlabel('SNR (dB)',loc='right')
        ax.set_ylabel('Altitude (km)',loc='top')
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0,1.15), frameon=False,ncol=3,borderaxespad=0.0,handletextpad=0.3)
        ax.set_xlim(-50, 50)
        plt.text(30,125,f'ZA$\in$({zas[zi]},{zas[zi+1]})')

        plt.show()
            
                    
            
        
