# -*- coding:utf-8 -*-
# Author: Yu Wang

import numpy as np
import matplotlib
matplotlib.use('Agg')
from read import *
import pandas as pd
import matplotlib.pyplot as plt
import sys,glob
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pylab as pylab
import itertools
import sklearn.decomposition as decomposition
from write import plot_rfi
from scipy.fftpack import fft
import fft_classfication


secperday = 3600 * 24
cindex_info_1=[]
cindex_1=[]
rfi_component_1=[]
rfi_basis_1=[]
f_info_1=[]
t_info_1=[]
t_info_1_=[]

cindex_info_2=[]
cindex_2=[]
rfi_component_2=[]
rfi_basis_2=[]
f_info_2=[]
t_info_2=[]

cindex_info_3=[]
cindex_3=[]
rfi_component_3=[]
rfi_basis_3=[]
f_info_3=[]
t_info_3=[]
data = []
time = []


# main code
if __name__ == '__main__':
   filepath = sys.argv[1]    #输入文件路径
   filelists = sorted(glob.glob(filepath+'*.mat'))
   for file in filelists:
      flux,f_arr = read_mat(file)
      t_end_str = read_time(file)
      data.append(flux)
      time.append(t_end_str)

   time = np.array(time).squeeze()
   time = [pd.to_datetime(t) for t in time]
   t_total = time[-1] - time[0]
   data = np.array(data).squeeze()
   l,m = data.shape
   basename = read_time(filelists[0])+'_'+read_time(filelists[-1])+'_'+str(int(f_arr[0]))+'_'+str(int(f_arr[-1]))
   bandpass = data.sum(axis=0)
   print('data_raw',data.shape)


   pca = decomposition.PCA(10)
   pca.fit(data)
   base = pca.transform(data)
   print('base:',base.shape)
   arr1 = np.array(pca.components_)
   print('component',arr1.shape)

   arr2 = np.array(base)

   l,m = arr1.shape
   mean = arr1.mean(axis=1).squeeze()
   std = arr1.std(axis=1).squeeze()

   for i in range(l):
      i = int(i)
      index = np.where((arr1[i,:]<mean[i]+3*std[i]) & (arr1[i,:]>mean[i]-3*std[i]))
      arr1[i,index]=0

   base = np.dot(data,arr1.T)
   print('base:',base.shape)


   weight_arr = [np.std(base[:,i])**2 for i in range(pca.n_components_)]   #时域干扰的标准差
   weight_arr_ratio = np.array(weight_arr)/sum(weight_arr)
   weight_sort = np.argsort(-weight_arr_ratio)
   arr1=np.array([arr1[i] for i in weight_sort])
   weight=weight_arr_ratio[weight_sort]
   base=np.array([base[:,i] for i in weight_sort])

   fig= plt.figure()
   params={
   'axes.labelsize': '10',
   'xtick.labelsize':'10',
   'ytick.labelsize':'10',
   'lines.linewidth':'0.3' ,
   'legend.fontsize': '10',
   }
   pylab.rcParams.update(params)

   t_arr = [t.strftime('%H:%M:%S') for t in time]
   t_arr1=t_arr

   plt.subplots_adjust(wspace=0.002)
   ax = fig.add_subplot(1, 2, 1)
   fig.autofmt_xdate()
   ax.set_xlabel('Time')
   ax.set_ylabel('Components')
   ax3 = fig.add_subplot(1, 2, 2)
   ax3.set_ylabel('Basis')
   ax3.set_xlabel('Frequency(MHz)')
   ax3.set_yticks([])
   ax3.yaxis.set_label_position("right")
   arr2=np.array(arr2)
   index=[]
   #显示权值大于0.005的干扰
   com_num = len(weight[weight>0.005])
   base=base[:com_num,:]
   arr1=arr1[:com_num,:]
   print('com_num',com_num)
   for j in range(com_num):
      y_arr1 = (arr1[j,:]-min(arr1[j,:]))/(max(arr1[j,:])-min(arr1[j,:]))+(20-j)
      ax3.plot(f_arr,y_arr1,linewidth=0.3)
      index0=list(np.array(np.where(abs(arr1[j])>7*np.std(arr1[j]))).reshape(-1))
      ax3.scatter(f_arr[index0],y_arr1[index0],s=0.2)
      index=index+index0
   c=sorted(list(set(np.array(index))))
   impulsive_count=-1
   for i in range(com_num):
      a= base[i]
      arr22 = np.array(np.array(arr1)[i])
      #####classify rfi #####
      rfi_type,data_fft = fft_classfication.classify_rfi(a)
      a = list(a)
      a = list((a-min(a))/(max(a)-min(a)))
      arr2 = list(arr2)
      if rfi_type=='Impulsive':
         impulsive_count+=1
         f_rfi = f_arr[list(arr22).index(max(arr22))]
         x_arr = np.arange(len(a))
         imp_index = x_arr[(a>=(5*np.std(a)+np.mean(a)))]
         imp_index_arr_output = fft_classfication.judge_continue(imp_index)
         imp_index_p =  [np.where(np.array(a)==max(np.array(a)[q]))[0][0] for q in imp_index_arr_output]
         t_rfi0=t_arr[imp_index_p]
         ###calculate stn of pulses###
         acopy = a[:]
         [acopy.remove(a[j]) for j in list(itertools.chain.from_iterable(imp_index_arr_output))]
         amean = np.mean(acopy)
         stn_pulse = [float(np.round(a[p]/amean,2)) for p in imp_index_p]
         t_rfi1 = list(t_rfi0.strftime('%Y-%m-%d %H:%M:%S'))
         t_rfi2 = sorted(set(t_rfi1),key=t_rfi1.index)
         imp_index_p = list(imp_index_p)
         t_rfi2 = list(t_rfi2)
         stn_pulse = list(stn_pulse)
         t_rfi = list([imp_index_p,t_rfi2,stn_pulse])
         if impulsive_count<1:
            cindex_info_1.append(weight[i])
            cindex_1.append(i)
            rfi_component_1.append(a)
            rfi_basis_1.append(list(arr22))
            f_info_1.append(f_rfi)
            t_info_1_.append(t_rfi2)
            t_info_1.append(t_rfi)
         else:
            if t_rfi2 not in t_info_1_:
               cindex_info_1.append(weight[i])
               cindex_1.append(i)
               rfi_component_1.append(a)
               rfi_basis_1.append(list(arr22))
               f_info_1.append(f_rfi)
               t_info_1_.append(t_rfi2)
               t_info_1.append(t_rfi)
            else:
                  wide_com_num=t_info_1_.index(t_rfi2)
                  cindex_info_1[wide_com_num]=cindex_info_1[wide_com_num]+weight[i]
                  if cindex_info_1[wide_com_num] >=0.05:
                     f_info_1[wide_com_num]='whole band'
      if rfi_type=='Periodic':
         f_rfi = f_arr[list(abs(arr22)).index(max(abs(arr22)))]
         #data_fft0=data_raw[:,f_rfi]
         data_fft1=a
         #data_fft1=data_fft0.reshape(len(data_fft0)//16,16).sum(axis=1)
         ef_min = 2/t_total
         ef_max = len(data_fft1)/t_total/2
         ef_x = np.linspace(ef_min,ef_max,len(data_fft1)//2)
         ef_x = np.round(ef_x,2)
         data_fft2=list(abs(fft(data_fft1))[40:len(data_fft1)//2])
         ef_x = ef_x[40:]
         t_rfi = ef_x[data_fft2==max(data_fft2)][0]
         cindex_info_2.append(weight[i])
         cindex_2.append(i)
         rfi_component_2.append(a)
         rfi_basis_2.append(list(arr22))
         f_info_2.append(f_rfi)
         t_info_2.append(t_rfi)
      if rfi_type=='Sporadic':
         f_rfi = f_arr[list(abs(arr22)).index(max(abs(arr22)))]
         cindex_info_3.append(weight[i])
         cindex_3.append(i)
         rfi_component_3.append(a)
         rfi_basis_3.append(list(arr22))
         f_info_3.append(f_rfi)
         t_info_3.append('None')
      a_norm = (a-min(a))/(max(a-min(a)))
      a_abs = abs(np.array(a))
      a_abs = a_abs.tolist()
      ax.plot(t_arr1, a_norm+(10-i),linewidth=0.3)
      ax.set_xticks(t_arr1[::len(t_arr1)//5])

   plt.savefig('%s.png'%basename,dpi=800)
   print('Impulse-like:%s'%(len(cindex_info_1)))
   print('Periodic:%s'%(len(cindex_info_2)))
   print('Colored-noise:%s'%(len(cindex_info_3)))


   cindex_info = cindex_info_1+cindex_info_2+cindex_info_3
   resort_index = np.argsort(cindex_info)[::-1]
   if len(cindex_info_1)==0:
      if len(cindex_info_2)!=0:
         base = np.vstack((np.array(rfi_basis_2),np.array(rfi_basis_3)))
         component = np.vstack((np.array(rfi_component_2),np.array(rfi_component_3)))
      else:
         base=np.array(rfi_basis_3)
         component = np.array(rfi_component_3)
   elif len(cindex_info_2)==0:
      if len(cindex_info_3)!=0:
         base = np.vstack((np.array(rfi_basis_1),np.array(rfi_basis_3)))
         component = np.vstack((np.array(rfi_component_1),np.array(rfi_component_3)))
      else:
         base=np.array(rfi_basis_1)
         component = np.array(rfi_component_1)
   elif len(cindex_info_3)==0:
      base = np.vstack((np.array(rfi_basis_1),np.array(rfi_basis_2)))
      component = np.vstack((np.array(rfi_component_1),np.array(rfi_component_2)))
   else:
      base = np.vstack((np.array(rfi_basis_1),np.array(rfi_basis_2),np.array(rfi_basis_3)))
      component = np.vstack((np.array(rfi_component_1),np.array(rfi_component_2),np.array(rfi_component_3)))

   base = [base[i] for i in resort_index]
   components = [component[i] for i in  resort_index]
   weights=[cindex_info[i] for i in  resort_index]
   np.savez("%s"%basename,bandpass=bandpass,rfi_channel=c,component=components,basis=base,weight=weights,time=time,freq=f_arr)
   plt.close('all')

   #### save rfi info ####
   np.savez("%s_Impulse-like"%basename,cindex_info=cindex_info_1,cindex=cindex_1,rfi_component=rfi_component_1,rfi_basis=rfi_basis_1,f_info=f_info_1,t_info=t_info_1)
   np.savez("%s_Periodic"%basename,cindex_info=cindex_info_2,cindex=cindex_2,rfi_component=rfi_component_2,rfi_basis=rfi_basis_2,f_info=f_info_2,t_info=t_info_2)
   np.savez("%s_Colored-noise"%basename,cindex_info=cindex_info_3,cindex=cindex_3,rfi_component=rfi_component_3,rfi_basis=rfi_basis_3,f_info=f_info_3,t_info=t_info_3)

   
