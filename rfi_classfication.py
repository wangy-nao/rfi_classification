import numpy as np
import heapq
from scipy.fftpack import fft



def detect_peaks(x, mph,n_peaks):
    x = np.atleast_1d(x).astype('float32')
    peaks_index=list(map(list(x).index, heapq.nlargest(n_peaks, x)))
    if len(peaks_index)<=0:
        print('no peaks found')
        return list(1)
    if min(peaks_index)<=mph or max(peaks_index)>=len(x)-mph:
       peaks_index.remove(min(peaks_index))
       return peaks_index
    else:
        peaks_index=list(np.delete(peaks_index,np.where(np.array(peaks_index)<mph)[0]))
        mph_arr=np.arange(-mph,mph+1)
        mph_arr=np.delete(mph_arr,mph)
        no_peaks=[]
        for i in peaks_index:
            dx=[x[i]-x[i+j] for j in mph_arr]
            if (np.array(dx)>0).all()==False:
               no_peaks.append(i)
            else:
               pass
        for i in no_peaks:
            peaks_index.remove(i)
        return peaks_index

def judge_continue(array):
    diff = array[1:]-array[:-1]
    right_edge = np.where(diff>50)[0]
    array_split = np.split(np.array(array),np.array(right_edge)+1)
    out_put = [list(array_split[i]) for i in range(len(array_split))]
    return out_put

### RFI classify and plot #####
def classify_rfi(data):
    ###detree1###
    data = (data-min(data))/(max(data)-min(data))
    std = np.std(data)
    median = np.median(data)
    data_fft = abs(fft(data))[1:len(data)//2]
    detree1=data[(data>=7*std+median)].size
    detree1 += data[(data<=median-7*std)].size
    #detree11 = max(data)/np.median(data)
    if detree1>=1:
       rfi_type='Impulsive'
       return rfi_type,data_fft
    else:
       ###detree2###
       peaks_index=list(map(list(data_fft).index, heapq.nlargest(15, data_fft)))
       detree2 = max(peaks_index)
       # 2020-10-13 更改30——>40
       if detree2 <=40:
          rfi_type='Sporadic'
          return rfi_type,data_fft
       else:
             ###detree3###
             peak_x = detect_peaks(data_fft,mph=50,n_peaks=15)
             mean50 = np.mean(data_fft[:50])
             mean_peaks = np.mean(data_fft[peak_x])
             peak_x_mean = np.mean(peak_x)
             detree3_1 = mean_peaks/np.mean(data_fft[50:])
             detree3_2 = mean_peaks/mean50
             detree3_3 = max(data_fft[50:])/mean50
             detree3 = detree3_1+detree3_2+detree3_3
            # 2020-10-13  10——>120
             if detree3 <= 160:
                rfi_type = 'Sporadic'
                return rfi_type,data_fft
             else:
                # 120——>480
                if peak_x_mean<480:
                   rfi_type = 'Sporadic'
                else:
                   std1 = np.std(data_fft[30:])
                # 10——>160
                   if std1*detree3<160:
                      rfi_type = 'Sporadic'
                   else:
                      rfi_type = 'Periodic'
                return rfi_type,data_fft

