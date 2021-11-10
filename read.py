from scipy.io import loadmat

def read_mat(file):
    data = loadmat(file)
    flux = data['data']
    freq = data['freq'].astype('float32').squeeze() #频率
    return (flux,freq)

def read_time(filename):
    data = loadmat(filename)
    t_end_str = data['t_end_str']
    time = t_end_str[0][:8]+t_end_str[0][9:15]
    return time