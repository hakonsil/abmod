import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import glob
import h5py #type: ignore
import scipy.interpolate as spi #type: ignore

path = '/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/'
abmod_files = glob.glob('/home/hakon/Documents/abmod/runs/run_ 500_*.txt')
m_add = 2
files = glob.glob(path + "*.h5")
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_0_30.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_30_60.h5')
files.remove('/home/hakon/Documents/meteor_fork/hakon/master/dl_analysed/new_profiles/snr10_anem0.05_c0.5/snr_za_60_90.h5')
print(files)
abmod_vs = [int(f[41+m_add:43+m_add]) for f in abmod_files]
abmod_zas = [int(float(f[45+m_add:47+m_add])) for f in abmod_files]
abmod_vs = np.array(abmod_vs)
abmod_zas = np.array(abmod_zas)
#print(abmod_files)
tf = files[3]
alt = np.arange(70, 131, 1)
fig = plt.figure(figsize=(16,10))
ax1 = fig.add_axes([0.05,0.1, 0.42, 0.8]) #drop
ax2 = fig.add_axes([0.55-0.02,0.1,0.42,0.8]) #nodrop
uncertainties = ["STD","SE","95% CI"]
#define 6 colors for plotting, going from light blue to dark blue
colors = ['#000aff','#004194','#003569','#002551','#00143a','#00020a']
with h5py.File(tf, 'r') as hf:
    keys = list(hf.keys())
    za1 = tf[tf.find("za_")+3:tf.find("za_")+5]
    if za1[-1] == '_':
        za1 = int(za1[0])
        za2 = int(tf[tf.find("za_")+5:tf.find("za_")+7])
    else:
        za1 = int(za1)
        za2 = int(tf[tf.find("za_")+6:tf.find("za_")+8])

    for i,key in enumerate(keys):
        drop = True
        if 'nodrop' in key:
            drop = False
        if drop:
            v1 = key[9:11]
            v2 = key[12:15]
        else:
            v1 = key[11:13]
            v2 = key[14:16]

        snr = hf[key][()]
        if v1 == '10':
            if drop:
                ax1.plot(10*np.log10(np.mean(snr, axis=0)), alt, label=f'{v1}_{v2}', color=colors[np.floor(int(v1)/10).astype(int)-1])
            else:
                ax2.plot(10*np.log10(np.mean(snr, axis=0)), alt, label=f'{v1}_{v2}', color=colors[np.floor(int(v1)/10).astype(int)-1])

    abmod_f = []
    abmod_v = []
    for i in range(len(abmod_files)):
        if abmod_zas[i] >= za1 and abmod_zas[i] <= za2:
            abmod_f.append(abmod_files[i])
            abmod_v.append(abmod_vs[i])
    #sort abmod_f and abmod_v by abmod_v
    abmod_f = [x for _, x in sorted(zip(abmod_v, abmod_f))]
    abmod_v = sorted(abmod_v)

    for i, af in enumerate(range(len(abmod_f))):
        afile = abmod_f[af]
        aalt, asnr = np.loadtxt(afile, unpack=True)
        aalt = aalt[::-1]
        asnr = asnr[::-1]
        aalt = aalt[100:701]
        asnr = asnr[100:701]

        f = spi.interp1d(aalt, asnr, kind='linear', fill_value='zero')
        int_snr = f(alt)
        print(i)
        if abmod_v[af] == 15:
            ax1.plot(int_snr, alt, label=abmod_v[af], linestyle='--', color=colors[i])
            ax2.plot(int_snr, alt, label=abmod_v[af], linestyle='--', color=colors[i])



        

        

    fig.suptitle(f'ZA: {tf[tf.find("za")+3:tf.find("za")+8]}')
ax1.set_xlabel('SNR [dB]')
ax1.set_ylabel('Altitude [km]')
ax1.set_title('Drop')
ax1.legend()
ax1.set_xlim(-50, 50)

ax2.set_xlabel('SNR [dB]')
ax2.set_ylabel('Altitude [km]')
ax2.set_title('No Drop')
ax2.legend()
ax2.set_xlim(-50, 50)
plt.show()



#plt.plot(10*np.log10(np.mean(drop_10_226, axis=0)), alt, label='rcs_drop_10.0_22.6')
#plt.show()





