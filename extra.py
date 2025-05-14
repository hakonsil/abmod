import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import glob
from itertools import chain
from tqdm import tqdm

def get_files():
    """
    Function to create a list of all .h5 files in /maarsy dir
    Parameters
    ----------
    test : bool, optional
        If True, the function will only return the first 10 files. The default is False.

    kep_collect : bool, optional
        If True, returns the fully processed kep files, if false returns the .h5 files on server
    Returns
    -------
    files : list
        List of all .h5 files in /maarsy dir except interstellar and perseids2023
    """
    
    yrs = glob.glob('/home/hakon/Documents/meteor_fork/data/*')
    yrs.remove('/home/hakon/Documents/meteor_fork/data/neutral_density')
    yrs.remove('/home/hakon/Documents/meteor_fork/data/obs_time.h5')

    mnts = [glob.glob(yr + '/*') for yr in yrs]
    mnts = list(chain(*mnts))


    files = [glob.glob(mnt + '/*.h5') for mnt in mnts]
    files = list(chain(*files))

    return files
files = get_files()
print(len(files))


bcs = np.array([])
vels = np.array([])
zas = np.array([])

for f in files:
    with h5py.File(f, 'r') as hf:
        try:
            bc = hf['bc'][:]
            vel = hf['vels'][:]
            za = hf['zenith_angle'][:]
            bcs= np.concatenate((bcs, bc))
            vels = np.concatenate((vels, vel))
            zas = np.concatenate((zas, za))

        except KeyError:
            print(f"KeyError: {f} does not contain 'bc' or 'vel' key.")
            continue


density = 1000
radii = bcs/density
volume = (4/3)*np.pi*radii**3
mass = density*volume*1e9

#remove nans
vels = vels[~np.isnan(mass)]
zas = zas[~np.isnan(mass)]
mass = mass[~np.isnan(mass)]

vbins = [0, 20, 30, 40, 50, 60, 100]
zbin = [0, 10, 30, 50, 70,90]
m001=[".01", ".02", ".03", ".04", ".05", ".06", ".07", ".08", ".09"]
m01=[".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
m1=["1", "2", "3", "4", "5", "6", "7", "8", "9"]
m10=["10", "20", "30", "40", "50", "60", "70", "80", "90"]
m100=["100", "200", "300", "400", "500", "600", "700", "800", "900"]
masses = m001 + m01 + m1 + m10 + m100
mb001 = [0, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095]
mb01 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
mb1 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
mb10 = [15, 25, 35, 45, 55, 65, 75, 85, 95]
mb100 = [150, 250, 350, 450, 550, 650, 750, 850, 1000]
mbins = mb001 + mb01 + mb1 + mb10 + mb100

print(len(mbins))
counts = np.zeros((6, 5, 46))
for i in tqdm(range(len(vels))):
    for j in range(6):
        if vels[i] >= vbins[j] and vels[i] < vbins[j+1]:
            for k in range(5):
                if zas[i] >= zbin[k] and zas[i] < zbin[k+1]:
                    for l in range(45):
                        if mass[i] >= mbins[l] and mass[i] < mbins[l+1]:
                            counts[j][k][l] += 1
with h5py.File('/home/hakon/Documents/abmod/counts.h5', 'w') as hf:
    hf.create_dataset('counts', data=counts)












"""
mass divided by vel
<20: 129.76744093245927
20-30: 70.53021042252533
30-40: 48.392138768685285
40-50: 35.3103003043128
50-60: 19.645068859234712
>60: 33.62726344245044

to do:
create a file where i can input mass velocity and zenith angle bins and get the number of meteors in each bin
"""


