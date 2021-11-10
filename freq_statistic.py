from os import startfile
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import sys,glob


filepath = sys.argv[1]
filelist = sorted(glob.glob(filepath+'*500.npz'))
start_freq = 1000
end_freq = 1500

for filename in filelist:
    print(filename)
    data = np.load(filename,allow_pickle=True)

    freq = data['freq']
    rfi_channels = data['rfi_channel']
    rfi_freq = freq[rfi_channels]

    #start_freq = np.min(freq[0],start_freq)
    #end_freq = np.max(freq[-1],end_freq)

    time = data['time']
    time = np.array(time).squeeze()
    time = [pd.to_datetime(t) for t in time]
    start = time[0]
    end = time[-1]
    duration = end-start

f_arr = np.arange(start_freq,end_freq,5)
rfi_ratio = np.zeros_like(f_arr)

fig = plt.figure()
for i in f_arr:
    rfi_num = len(rfi_freq[rfi_freq>i-2.5 & rfi_freq<i+2.5])
    rfi_ratio[i] = rfi_num/duration
plt.hist(rfi_ratio)

plt.savefig('rfi_ratio'+'_freq.png',dpi=400)

