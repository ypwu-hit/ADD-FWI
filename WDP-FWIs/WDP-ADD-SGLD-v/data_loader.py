# @author: wuyuping (ypwu@stu.hit.edu.cn)

import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from scipy.signal import butter
from torchaudio.functional import biquad

def DataLoad_Train(n_shots):

    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    # device = torch.device('cuda:0')
    # device = torch.device('cpu')

    data_path = '../../../data/'
    
    ny = 340
    nx = 130

    v_true = torch.from_file(data_path+'mar_big_vp_130_340.bin',
                        size=ny * nx).reshape(ny, nx)

    dx = 22.5 # 35
    
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
    d_source = 11  # 10 * 8m = 80m
    first_source = 5  # 5 * 8m = 40m
    source_depth = 1  # 1 * 8m = 8m

    n_receivers_per_shot = 339
    d_receiver = 1  # 3 * 8m = 24m
    first_receiver = 0  # 0 * 8m = 0m
    receiver_depth = 1  # 1 * 8m = 8m

    freq = 7
    nt = 2000
    dt = 0.002
    peak_time = 1.0 / freq

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
            .repeat(n_shots, 1)
    )

    # source_amplitudes [shot, source, time]
    # source_amplitudes [shot, source, time]
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    )

    sos = butter(6, 5, 'hp', fs=1/dt, output='sos')
    sos = [torch.tensor(sosi).to(source_amplitudes.dtype)
           for sosi in sos]

    def filt(x):
        return biquad(biquad(biquad(x, *sos[0]), *sos[1]),
                      *sos[2])

    source_amplitudes_filt = filt(source_amplitudes).repeat(n_shots, n_sources_per_shot, 1)

    # Propagate [num_shots, n_r, n_t]
    # observed_data = scalar(v_true, dx, dt, source_amplitudes=source_amplitudes,
    #                        source_locations=source_locations,
    #                        receiver_locations=receiver_locations,
    #                        accuracy=8,
    #                        pml_freq=freq)[-1]

    # Propagate [num_shots, n_r, n_t] observed_data
    observed_data = (
    torch.from_file(data_path+'marmousi2_130_340_data_experiments3_3_filt.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
    )
    print('observed_data.shape', observed_data.shape)
    data_dim = observed_data[0].reshape

    noise = (
    torch.from_file(data_path+'marmousi2_130_340_data_noise_experiments3_3_filt.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
    )
    print('noise.shape', noise.shape)
    
    observed_data_noise = observed_data + noise

    return observed_data, observed_data_noise, data_dim, model_dim, source_amplitudes_filt, source_locations, receiver_locations, v_true

# DataLoad_Train(30)