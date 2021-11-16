# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import math
import sys,glob
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


filename='./20190626144207_20190626151742_50_500.npz'
#filepath = sys.argv[1]
#filenames = sorted(glob.glob(filepath+'*.npz'))
name = str(filename).replace(".npz"," ")
data0 = np.load(filename,allow_pickle=True)
print(data0.files)

freq = data0['freq']

rfi_ratio = len(data0['rfi_channel'])/len(freq)
ratio='%.2f'%(rfi_ratio*100)+'%'

bandpass = data0['bandpass']

time = data0['time']
time = np.array(time).squeeze()
time = [pd.to_datetime(t) for t in time]
start = time[0]
end = time[-1]
duration = end-start

az = '-'
el = '-'
weight = data0['weight']
tot_num=len(weight)
com_num = min(10,len(weight))
weight = weight[:com_num]

weight=[float('{0:.4f}'.format(100*i)) for i in weight]
print('rfi_channel:',len(data0['rfi_channel']))
com = data0['component']
print('com:',com.shape)
base = data0['basis']
com_scale = int(data0['component'].shape[1])
print(com_scale)

t_arr2 = [t.strftime('%H:%M:%S') for t in time]
fig= plt.figure(figsize=(20, 12))
params={'axes.labelsize': '15',
                'xtick.labelsize':'15',
                'ytick.labelsize':'15',
                'lines.linewidth':'0.5' ,
                'legend.fontsize': '15',}
pylab.rcParams.update(params)
font={'family':'serif',
      'style':'normal',
      'color':'black',
      'size':13}

### 显示文件相关信息
ax11=fig.add_axes([0.05, 0.8, 0.8,0.15])
ax11.axis('off')
ax11.text(0,1.05,'FILE:',fontdict=font,fontsize=15)
ax11.text(0.05,1.05,name,fontsize=15,fontweight='light')
ax11.text(0.0,0.8, 'AZ:',fontdict=font,fontsize=15)
ax11.text(0.05,0.8, az,fontsize=15,fontweight='light')
ax11.text(0.0,0.55,'EL:',fontdict=font,fontsize=15)
ax11.text(0.05,0.55, el,fontsize=15,fontweight='light')
ax11.text(0.,0.3, 'TELESCOPE:',fontdict=font,fontsize=15)
ax11.text(0.115,0.3, '4.5m Satellite Antenna',fontsize=15,fontweight='light')
ax11.text(0.0,0.05,'Duration:',fontdict=font,fontsize=15)
ax11.text(0.106,0.05,'%s'%(duration),fontsize=15,fontweight='light')
ax11.text(0.40,0.05,'RFI RATIO:',fontdict=font,fontsize=15)
ax11.text(0.53,0.05,'%.2f%%'%(100*rfi_ratio),fontsize=15,weight='light')
ax11.text(0.40,1.05, 'UTC:',fontdict=font,fontsize=15)
ax11.text(0.45,1.05, t_arr2[0],fontsize=15,weight='light')
ax11.text(0.40,0.8,'RFI TYPES:',fontdict=font,fontsize=15)
ax11.text(0.50,0.8,'%s;'%('Impulsive'),fontsize=15,weight='light',color='red')
ax11.text(0.60,0.8,'%s'%('Non-stationary'),fontsize=15,weight='light',color='deepskyblue')
ax11.text(0.40,0.55,'CENTERY FREQUENCY:',fontdict=font,fontsize=15)
ax11.text(0.62,0.55, '%s MHz'%(int((freq[-1]+freq[0])/2)),fontsize=15,weight='light')
ax11.text(0.40,0.3,'COMPONENT NUMBER:',fontdict=font,fontsize=15)
ax11.text(0.63,0.3, '%s'%tot_num,fontsize=15,weight='light')

### 作图
ax22=fig.add_axes([0.76, 0.83, 0.2,0.15])
ax22.plot(freq,bandpass,'black',label='Bandpass (MHz)')
ax22.scatter(freq[data0['rfi_channel']],bandpass[data0['rfi_channel']],c='red',marker='x',s=15,label='RFI Channel')
ax22.tick_params(axis="x", labelsize=15)
ax22.legend(prop={'size': 10},framealpha=0.9)
ax22.set_yticks([])


sporad_data = np.load(filename.replace('.npz','_Colored-noise.npz'))
imp_data = np.load(filename.replace('.npz','_Impulse-like.npz'),encoding='latin1')
va_imp = imp_data['cindex_info']
va_imp_sum = sum(va_imp)
va_sporad = sporad_data['cindex_info']
va_sporad_sum = sum(va_sporad)
va = list(va_imp)+list(va_sporad)
va = np.sort(np.array(va))[::-1]
va_sum = sum(va)
weights=[va_imp_sum/va_sum,va_sporad_sum/va_sum]
ax33=fig.add_axes([0.57, 0.80, 0.21,0.15])
ax33.tick_params(axis="y", labelsize=15)
ax33.set_title('Variance-Ratio of RFI types', fontdict = font,fontsize=10)
explode = [0,0]
colors=['red','deepskyblue']
ax33.pie(weights,colors=colors,explode=explode,autopct='%1.1f%%',shadow=False,startangle=250)

fig.subplots_adjust(hspace=0,wspace=0,left = 0.05,right = 0.96,bottom = 0.05,top = 0.8)
left, width = 0.05, 0.42
bottom, height = 0.12, .68
bottom_m = left_m = left+width+0.0
bottom_r = left_r = left+width+0.07
rect_left = [left, bottom, width, height]
rect_mid = [left_m, bottom, 0.07, height]
rect_right = [left_r,bottom,0.42,height]
ax=plt.axes(rect_left)
ax2=plt.axes(rect_right)
ax1=plt.axes(rect_mid)
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('RFI Strength',fontsize=15)
ax.set_yticks([])
ax1.set_xlabel(r'Ratio (e^)',fontsize=15)
ax1.set_yticks([])
ax1.set_xticks([])
colors = ['#1a55FF','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

weight_y=np.arange(com_num)[::-1]
ax1.plot(np.log(weight),weight_y,'-.',linewidth=1,color='black')
for i in range(com_num):
    ax1.scatter(np.log(weight[i]),weight_y[i],s=50,c=colors[i%10])
for sca_x, sca_y in zip(np.log(weight),weight_y):
    if sca_x>=max(np.log(weight))*0.8:
       print(sca_x)
       sca_x_=sca_x-0.3*max(np.log(weight))
    elif sca_x<=0.2*max(np.log(weight)):
       sca_x_=sca_x+0.3*max(np.log(weight))
    else:
       sca_x_=sca_x 
    ax1.annotate('%.2f%%'%(math.e**sca_x),xy=(sca_x_, sca_y),xytext=(0,-5),
                 textcoords='offset points',ha='center',va='top',fontsize=15) 


print('t_arr2:',len(t_arr2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=-20)
for i in range(com_num):
    comi=com[i]
    x1 = (comi-min(comi))/(max(comi)-min(comi))
    ax.plot(t_arr2,x1+com_num-i,color=colors[i%10])
    ax.set_xticks(t_arr2[::len(t_arr2)//5])

### the RFI Bases - Freq figure ###
ax2.set_yticks([])
ax2.set_xlabel('Frequency (MHz)',fontsize=15)
ax2.set_ylabel('RFI Bases',fontsize=15)
ax2.yaxis.set_label_position("right")
for i in range(com_num):
    x2 = (base[i]-min(base[i]))/(max(base[i])-min(base[i]))
    index = np.arange(len(x2))[(np.array(x2)>=3*np.std(x2)+np.mean(x2))|(np.array(x2)<=-3*np.std(x2)+np.mean(x2))]
    ax2.plot(freq,x2+com_num-i,linewidth=1.5,color=colors[i%10])
plt.savefig(filename.replace('.npz','_new.png'),dpi=400)
plt.close()

