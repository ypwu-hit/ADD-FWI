# @author: wuyuping (ypwu@stu.hit.edu.cn)

import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import scipy
import torch.nn.functional as F

def sliding_average_filter(signal, size=5, std=1, mode='gau'):
    """
    滑动平均滤波器
    :param size: 滤波器的大小
    :param std: 控制标准差，决定滤波器的分布
    :return: 滑动平均滤波器的PyTorch张量
    """
    if mode == 'gau':
        # 创建一个1D滤波器的权重，使用高斯分布
        weights = torch.exp(-(torch.arange(size).float() - (size - 1) / 2) ** 2 / (2 * std ** 2))
    else:
        weights = torch.ones(size)
    
    # 归一化权重，确保总和为1
    weights /= weights.sum()
    # 创建1D卷积层，设置卷积核为创建的滑动平均滤波器

    return F.conv1d(F.pad(signal.view(1,1,-1),((size - 1) // 2,(size - 1) // 2), mode='replicate'), weights.view(1, 1, size), 
                                      bias=None, stride=1, padding=0)[0,0]

# 高通滤波
def filter_highpass(signal, low_l, low_cut, nt, dt):

    # 傅里叶变换
    signal_fft = torch.fft.rfft(signal)
    # 滤波系数, low_l左端点, low_cut右端点
    co_e = co_filter_highpass(low_l, low_cut, nt, dt, signal_fft.shape[-1])

    signal_fft_filtered = (signal_fft * co_e)

    signal_filtered = torch.fft.irfft(signal_fft_filtered)

    return signal_filtered

# 高通滤波系数
def co_filter_highpass(low_l, low_cut, nt, dt, nt_fft):
    #     signal_fft = torch.fft.rfft(signal)

    co_signal = torch.ones(nt_fft)

    # freq = torch.arange((n + 1) // 2) / (d * n)
    # index = freq * (d * n)

    # end
    low_cut_index = int(low_cut * dt * nt)
    # begin
    low_l_index = int(low_l * dt * nt)

    co_signal[0:low_l_index + 1] = 0

    # tensor([0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750])
    co_signal[low_l_index:low_cut_index] = torch.arange(0, 1, 1.0 / (low_cut_index - low_l_index))

    # 分段平滑
    index_length = 4
    co_signal[low_cut_index-index_length:low_cut_index+index_length] = torch.arange(co_signal[low_cut_index-index_length], 
                                                                                    1, (1-co_signal[low_cut_index-index_length]) / (2*index_length))
    
    low_cut_index = low_cut_index + index_length
    index_length = 3
    co_signal[low_cut_index-index_length:low_cut_index+index_length] = torch.arange(co_signal[low_cut_index-index_length], 
                                                                                    1, (1-co_signal[low_cut_index-index_length]) / (2*index_length))
    
    low_cut_index = low_cut_index + index_length
    index_length = 2
    co_signal[low_cut_index-index_length:low_cut_index+index_length] = torch.arange(co_signal[low_cut_index-index_length], 
                                                                                    1, (1-co_signal[low_cut_index-index_length]) / (2*index_length))
    
    
    
    index_length = 4
    co_signal[low_l_index-index_length:low_l_index+index_length] = torch.arange(0, co_signal[low_l_index+index_length], 
                                                                                co_signal[low_l_index+index_length] / (2*index_length))
    
    low_l_index = low_l_index - index_length
    index_length = 3
    co_signal[low_l_index-index_length:low_l_index+index_length] = torch.arange(0, co_signal[low_l_index+index_length], 
                                                                                co_signal[low_l_index+index_length] / (2*index_length))
    
    low_l_index = low_l_index - index_length
    index_length = 2
    co_signal[low_l_index-index_length:low_l_index+index_length] = torch.arange(0, co_signal[low_l_index+index_length], 
                                                                                co_signal[low_l_index+index_length] / (2*index_length))

    return sliding_average_filter(co_signal)
    # return co_signal

def DataLoad_Train(n_shots):

    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    # device = torch.device('cuda:0')
    # device = torch.device('cpu')

    data_path = '../../data/'
    
    ny = 340
    nx = 130

    v_true = torch.from_file(data_path+'mar_big_vp_130_340.bin',
                        size=ny * nx).reshape(ny, nx)
    v_true = v_true[70:-70,:]

    dx = 12.5 # 35
    
    Vvmax = torch.max(v_true)
    Vvmin = torch.min(v_true)
    print('vmin, vmax', Vvmin, Vvmax)

    model_dim = v_true.shape
    print('v_true.shape', v_true.shape)

    plt.figure()
    plt.imshow(v_true.T, aspect='auto', cmap='jet')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Velocity")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('example_v.png')
    plt.close()

    n_shots = n_shots
    
    n_sources_per_shot = 1
    d_source = 10  # 10 * 8m = 80m
    first_source = 4  # 5 * 8m = 40m
    source_depth = 2  # 1 * 8m = 8m

    n_receivers_per_shot = 200
    d_receiver = 1  # 3 * 8m = 24m
    first_receiver = 0  # 0 * 8m = 0m
    receiver_depth = 3  # 1 * 8m = 8m

    freq = 15
    nt = 4000
    dt = 0.001
    peak_time = 6.0 / freq

    print(deepwave.common.cfl_condition(dy = dx, dx = dx, dt = dt, max_vel = 4700))

    # source_locations, [shot, source, space]
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

    # receiver_locations [shot, receiver, space]
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.long)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver)
               .repeat(n_shots, 1))

    # 拖缆设置
    for index in range(n_shots):
        # 首先判断这个长度来判断拖缆的方向, 
        # 如果炮点位置(source_locations[index,0,0])后延长拖缆距离(80 + 24)不超出则可以设定
        if source_locations[index,0,0] + 80 + 24 < 200:
            begin = source_locations[index,0,0] + 4
            end = source_locations[index,0,0] + 4 + 76
            receiver_locations[index,0:begin,0] = 0
            receiver_locations[index,end:,0] = 0
        else:
            begin = source_locations[index,0,0] - 4
            end = source_locations[index,0,0] - 4 - 76
            receiver_locations[index,begin+1:,0] = 0
            receiver_locations[index,0:end+1,0] = 0

    n_receivers_per_shot_tuolan = torch.sum(torch.where(receiver_locations[:,:,0]>0,1,0),1)
    # receiver_locations [shot, receiver, space]
    receiver_locations_tuolan = torch.zeros(n_shots, n_receivers_per_shot_tuolan[0], 2,
                                     dtype=torch.long)
    receiver_locations_tuolan[..., 1] = receiver_depth
    receiver_locations_tuolan[:, :, 0] = (
        (torch.arange(n_receivers_per_shot_tuolan[0]) * d_receiver + first_receiver)
               .repeat(n_shots, 1))

    for index in range(n_shots):
        # 相当于找到检波器的位置索引
        index_receivers = torch.argwhere(receiver_locations[index,:,0] > 0)
        receiver_locations_tuolan[index, :, 0] = receiver_locations[index, index_receivers, 0][:,0]

    # source_amplitudes [shot, source, time]
    # source_amplitudes [shot, source, time]
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    )
    source_amplitudes_filtered = filter_highpass(source_amplitudes, 12, 13, nt, dt)
    filtered_highpass_f = source_amplitudes_filtered.repeat(n_shots, n_sources_per_shot, 1)


    # Propagate [num_shots, n_r, n_t]
    # observed_data = scalar(v_true, dx, dt, source_amplitudes=source_amplitudes,
    #                        source_locations=source_locations,
    #                        receiver_locations=receiver_locations,
    #                        accuracy=8,
    #                        pml_freq=freq)[-1]

    # Propagate [num_shots, n_r, n_t] observed_data
    observed_data = (
    torch.from_file(data_path+'marmousi2_130_200_data_experiments_filter_12_13_fs.bin',
                    size=n_shots*n_receivers_per_shot_tuolan[0]*nt)
    .reshape(n_shots, n_receivers_per_shot_tuolan[0], nt)
    )
    print('observed_data.shape', observed_data.shape)
    data_dim = observed_data[0].reshape

    noise = (
    torch.from_file(data_path+'marmousi2_130_200_data_noise_experiments_filter_12_13_fs.bin',
                    size=n_shots*n_receivers_per_shot_tuolan[0]*nt)
    .reshape(n_shots, n_receivers_per_shot_tuolan[0], nt)
    )
    print('noise.shape', noise.shape)
    
    observed_data_noise = observed_data + noise

    return observed_data, observed_data_noise, data_dim, model_dim, filtered_highpass_f, source_locations, receiver_locations_tuolan, v_true

# DataLoad_Train(30)