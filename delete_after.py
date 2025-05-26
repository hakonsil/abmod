# copy the file /nfs/revontuli/data/juha/maarsy/2020/03/20200319_220733182_event.ud3.h5best_fit.png to this directory

import os
import glob
import shutil

file = '/nfs/revontuli/data/juha/maarsy/2020/03/20200319_220733182_event.ud3.h5best_fit.png'
shutil.copy(file, '/home/hakon/Documents/abmod/20200319_220733182_event.ud3.h5best_fit.png')

file1 = '/nfs/revontuli/data/juha/maarsy/2020/03/20200319_173254184_event.ud3.h5best_fit.png'
file2 = '/nfs/revontuli/data/juha/maarsy/2020/03/20200319_204946012_event.ud3.h5best_fit.png'
file3='/nfs/revontuli/data/juha/maarsy/2020/03/20200319_204946012_event.ud3.h5best_fit.png'
file4='/nfs/revontuli/data/juha/maarsy/2020/06/20200602_044356240_event.ud3.h5best_fit.png'
"""
0.06269997358322144
/nfs/revontuli/data/juha/maarsy/2020/03/20200319_173254184_event.ud3.h5best_fit.png

0.722292959690094
/nfs/revontuli/data/juha/maarsy/2020/03/20200319_204946012_event.ud3.h5best_fit.png

0.722292959690094
/nfs/revontuli/data/juha/maarsy/2020/03/20200319_204946012_event.ud3.h5best_fit.png

/nfs/revontuli/data/juha/maarsy/2020/06/20200602_044356240_event.ud3.h5best_fit.png

"""
shutil.copy(file1, '/home/hakon/Documents/abmod/20200319_173254184_bf_062699.png')
shutil.copy(file2, '/home/hakon/Documents/abmod/20200319_204946012_bf_722292.png')
shutil.copy(file3, '/home/hakon/Documents/abmod/20200319_204946012_bf_722292.png')
shutil.copy(file4, '/home/hakon/Documents/abmod/20200602_044356240_bf.png')