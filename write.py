import matplotlib.pyplot as plt
import numpy as np



####plot rfi ####
def plot_rfi(t_data,t_info,f_info,cindex_info,rfi_type,t_text,f_text,t_arr1,basename):
    t_data = np.array(t_data)
    rfi_num = len(cindex_info)
    if rfi_num==0:
        pass
    elif rfi_num==1:
        fig, axs = plt.subplots(1,2, sharex=False)
        fig.suptitle('%s'%rfi_type,y=0.95,fontsize=20)
        fig.subplots_adjust(hspace=0,wspace=0)
        fig.autofmt_xdate()
        axs[0].plot(t_arr1,t_data[0],color='black')
        axs[0].set_yticks([np.max(t_data[0])/2])
        axs[0].set_xticks(t_arr1[::len(t_arr1)//8])
        axs[0].set_yticklabels(['%.2e'%cindex_info[0]],rotation=90)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        t_note = str("%s%s"%(t_text,t_info)).replace("\n"," ").replace("[","").replace("]","")
        f_note = str("%s%s"%(f_text,f_info)).replace("\n"," ").replace("[","").replace("]","")
        axs[1].text(0.005,0.95,t_note,ha='left',va='top',fontsize=5,wrap=True,transform=axs[1].transAxes)
        axs[1].text(0.005,0.7,f_note,fontsize=5,ha='left',va='top',wrap=True,transform=axs[1].transAxes)
        plt.savefig('%s_%s.png'%(basename,rfi_type),dpi=800)
        plt.close('all')
    elif rfi_num>1:
        fig, axs = plt.subplots(rfi_num,2, sharex=False)
        fig.suptitle('%s'%rfi_type,y=0.95,fontsize=20)
        fig.subplots_adjust(hspace=0,wspace=0)
        fig.autofmt_xdate()
        for i in range(rfi_num):
    # Plot each graph, and manually set the y tick values
            t_note = str("%s%s"%(t_text,t_info[i])).replace("\n"," ").replace("[","").replace("]","")
            f_note = str("%s%s"%(f_text,f_info[i])).replace("\n"," ").replace("[","").replace("]","")
            axs[i,0].plot(t_arr1,t_data[i],color='black')
    #        axs[i,0].plot(t_arr,np.zeros(len(t_data[i]))+np.mean(t_data[i])+5*np.std(t_data[i]))
            axs[i,0].set_yticks([np.max(t_data[i])/2])
            axs[i,0].set_xticks(t_arr1[::len(t_arr1)//8])
            axs[i,0].set_yticklabels(['%.2e'%cindex_info[i]],rotation=-60)
            axs[i,1].set_xticks([])
            axs[i,1].set_yticks([])
            axs[i,1].spines['right'].set_visible(False)
            axs[i,1].spines['top'].set_visible(False)
            axs[i,1].text(0.005,0.95,t_note,fontsize=5,wrap=True,ha='left',va="top",transform=axs[i,1].transAxes)
            axs[i,1].text(0.005,0.7,f_note,fontsize=5,wrap=True,ha='left',va="top",transform=axs[i,1].transAxes)
    #        plt.tight_layout()
        plt.savefig('%s_%s.png'%(basename,rfi_type),dpi=800)
        plt.close('all')
