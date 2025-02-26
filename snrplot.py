import numpy as n
import matplotlib.pyplot as plt
import glob
import re


masses=[".01",
        ".100000003431119",
        "1.00000006862238",
        "10.0000010293358",
        "100",
        "10",
        ".1",
        "1",
        "1.58489320333703E-2",
        ".15848932577166",
        "1.58489331209618",
        "15.8489336647576",
        "2.51188646598391E-2",
        ".251188655216973",
        "2.51188663835555",
        "25.1188672454137",
        "3.98107178749216E-2",
        ".398107192408748",
        "3.9810720606828",
        "39.8107219727813",
        "6.30957361799312E-2",
        ".630957383448211",
        "6.30957405097112",
        "63.0957426746012"]

vels=["15",
      "25",
      "35",
      "45",
      "55",
      "65"]

entry_angles=[
    "0",
    "20",
    "35",
    "40",
    "60",
    "80"
    ]
    
def plot_sweep(mass=".01",angle="40"):
    fl=glob.glob("run_ %s_* %s.txt"%(mass,angle))
    fl.sort()
    if len(fl) < 2:
        print("only found %d runs. not plotting"%(len(fl)))
        return
    vels=n.array([15,25,35,45,55,65],dtype=n.int64)
    n_vels=len(vels)
    SNR=n.zeros([1402,n_vels])
    hgts=[]
    for f in fl:
        print(f)
        a=n.genfromtxt(f)
        vel=int(re.search("run_.*_(.*)_ (.*).txt",f).group(1))
        vi=n.where(vel == vels)[0][0]
        print(a.shape)
        print(vi)
        SNR[:,vi]=a[:,1]
        if len(hgts)==0:
            hgts=a[:,0]
        print(a.shape)

        plt.plot(a[:,1],a[:,0],label="%s (km/s)"%(vel))
    plt.legend()
    plt.axvline(-7)
    plt.title(r"m=%1.3f $\mu$g $\alpha=%d^{\circ}$"%(float(mass),int(angle) ))
    plt.xlim([-200,80])
    plt.ylim([60,130])    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Height (km)")
    plt.savefig("plt-%s-%s.png"%(mass,angle))
    plt.close()
#   plt.show()

    if False:
        plt.pcolormesh(vels,hgts,SNR)
        plt.title(r"m=%1.3f $\mu$g $\alpha=%d^{\circ}$"%(float(mass),int(angle) ))
        plt.xlabel("Velocity (km/s)")
        plt.ylabel("Height (km)")
        cb=plt.colorbar()
        cb.set_label("SNR (dB)")
        plt.show()


for m in masses:
    for a in entry_angles:
        plot_sweep(mass=m,angle=a)
        
