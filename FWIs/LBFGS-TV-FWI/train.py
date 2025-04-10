# @author: wuyuping (ypwu@stu.hit.edu.cn)

import time
import torch
import numpy as np
from data_loader import DataLoad_Train
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from deepwave import scalar
import scipy.io
from scipy.ndimage import gaussian_filter
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
import random

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return (super(SSIM_Loss, self).forward(img1, img2) )

class MyDataset(data_utils.Dataset):
    def __init__(self,seismic_data, source_amplitudes, source_locations, receiver_locations):

        self.seismic_data = seismic_data
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

    def __getitem__(self, item):

        seismic_data = self.seismic_data[item]
        source_amplitudes = self.source_amplitudes[item]
        source_locations = self.source_locations[item]
        receiver_locations = self.receiver_locations[item]

        return seismic_data, source_amplitudes, source_locations, receiver_locations

    def __len__(self):
        return len(self.seismic_data)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

def ATVLoss(x):    
    """Compute L1-based TV loss for x

    Args:
        x: image, torch.Variable of torch.Tensor
    Returns:
           ATV loss

    alpha_tv * (ATVLoss(model))/(ny*nz)

    """

    dh       = x[:,1:] - x[:,:-1]
    dw       = x[1:,:] - x[:-1,:]
    atvloss  = torch.sum(torch.abs(dh[:-1,:]) + torch.abs(dw[:, :-1]))

    return atvloss

same_seeds(1234)

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

BatchSize = 6
number_of_shots = 30
dx = 22.5
dt = 0.002
freq = 7

seismic_data, seismic_data_noise, data_dim, model_dim, source_amplitudes, x_s, x_r, v_true = DataLoad_Train(number_of_shots)
# print('source_amplitudes', source_amplitudes.shape, x_s.shape, x_r.shape)

model_max = torch.max(v_true)
model_min = torch.min(v_true)
v_true_nor = (v_true - model_min) / (model_max - model_min)
v_true_nor = v_true_nor.to(device)

train_loader = data_utils.DataLoader(MyDataset(seismic_data_noise, source_amplitudes, x_s, x_r),
                                     batch_size=BatchSize,
                                     shuffle=True, drop_last=False)

v_init = (torch.tensor(1/gaussian_filter(1/v_true.cpu().numpy(), 30)).to(device))
v_true = v_true.to(device)

plt.figure()
plt.imshow(v_init.detach().cpu().numpy().T, aspect='auto', cmap='jet', vmax=4700, vmin=1100)
plt.xlabel("X")
plt.ylabel("Z")
plt.title("v_init")
plt.colorbar()
plt.tight_layout()
plt.savefig('v_init.png')
plt.close()
# v = v_init.clone()
# v.requires_grad_()

class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) /
                        (max_vel - min_vel))
        )

    def forward(self):
        return (torch.sigmoid(self.model) *
                (self.max_vel - self.min_vel) +
                self.min_vel)


model = Model(v_init, 1100, 4700).to(device)


# Setup optimiser to perform inversion
# optimizer = torch.optim.LBFGS([v], lr=10, 
#     line_search_fn='strong_wolfe', 
#     max_iter=100,
#     history_size=7)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1,
                                  line_search_fn='strong_wolfe',
                                  max_iter=150, history_size=7)
# optimizer = torch.optim.Adam([v], lr=25, eps=1e-18)
loss_fn = torch.nn.MSELoss()
loss_ssim = SSIM_Loss(data_range=1.0, size_average=True, channel=1)

print()
print('*******************************************')
print('*******************************************')
print('           START TRAINING                  ')
print('*******************************************')
print('*******************************************')
print()

DisplayStep = 2
loss_data_file = 'loss_data.txt'
loss_model_file = 'loss_model.txt'
loss_ssim_file = 'loss_ssim.txt'
step   = int(number_of_shots/BatchSize)
num_epochs = 600
start  = time.time()
# alpha_tv = 1
alpha_tv = 500

for epoch in range(num_epochs):

    since = time.time()
    # epoch_loss = 0.0

    def closure():
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, (s_d_i, s_a_i, x_s_i, x_r_i) in enumerate(train_loader):
            v_temp = model()
            
            # Forward prediction
            out = scalar(
                v_temp, dx, dt,
                source_amplitudes=s_a_i.to(device),
                source_locations=x_s_i.to(device),
                receiver_locations=x_r_i.to(device),
                max_vel=4700,
                pml_freq=freq,
                accuracy=8,
                pml_width=[20, 20, 0, 20]
            )[-1]

            loss = (
                1e6*loss_fn(out.to(device),
                        s_d_i.to(device)) + alpha_tv * (ATVLoss(v_temp))/(model_dim[0]*model_dim[1])
            )

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            epoch_loss += loss.item()
            # Loss backward propagation
            loss.backward()

            
        return epoch_loss


    optimizer.step(closure)
    
    # val
    with torch.no_grad():  # operations inside don't track history
        # Validation Mode:
        v_temp = model()
        observed_data_fd = scalar(v_temp.to(device), dx, dt,
                                  source_amplitudes=source_amplitudes.to(device),
                                  source_locations=x_s.to(device),
                                  receiver_locations=x_r.to(device),
                                  max_vel=4700,
                                  pml_freq=freq,
                                  accuracy=8,
                                  pml_width=[20, 20, 0, 20])[-1]

        loss_model = loss_fn(v_temp, v_true) / loss_fn(v_true, torch.zeros_like(v_true))

        if epoch == 0:
            loss_model_count = loss_model.detach().cpu()
            v_save = v_temp.detach().cpu().numpy()
        else:
            if loss_model.detach().cpu() < loss_model_count:
                loss_model_count = loss_model.detach().cpu()
                v_save = v_temp.detach().cpu().numpy()

        loss_data = loss_fn(observed_data_fd, seismic_data.to(device)) / loss_fn(seismic_data.to(device), 
            torch.zeros_like(seismic_data.to(device)))

        model_max = torch.max(v_temp)
        model_min = torch.min(v_temp)
        model_nor = (v_temp - model_min) / (model_max - model_min)
        loss_model_ssim = loss_ssim(model_nor.view(1, 1, model_dim[0], model_dim[1]),
                                    v_true_nor.view(1, 1, model_dim[0], model_dim[1]))

        if epoch == 0:
            loss_model_sum = loss_model.detach().cpu()
            loss_data_sum = loss_data.detach().cpu()
            loss_model_ssim_sum = loss_model_ssim.detach().cpu()
        else:
            loss_model_sum = np.append(loss_model_sum, loss_model.detach().cpu())
            loss_data_sum = np.append(loss_data_sum, loss_data.detach().cpu())
            loss_model_ssim_sum = np.append(loss_model_ssim_sum, loss_model_ssim.detach().cpu())

        with open(loss_data_file, 'a') as fr:
            fr.write(str(loss_data.detach().cpu().numpy()) + '\n')
        with open(loss_model_file, 'a') as fr:
            fr.write(str(loss_model.detach().cpu().numpy()) + '\n')
        with open(loss_ssim_file, 'a') as fr:
            fr.write(str(loss_model_ssim.detach().cpu().numpy()) + '\n')

        if epoch % 2 == 0:

            # torch.save(net.state_dict(), str(epoch)+'_model_parameters.pkl')

            plt.figure()
            plt.imshow(v_temp.detach().cpu().numpy().T, aspect='auto', cmap='jet', vmax=4700, vmin=1100)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title("Velocity")
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(str(epoch) + '.png')
            plt.close()

            v_temp.detach().cpu().numpy().tofile(str(epoch) + '_lbfgs_marmousi2_inv.bin')


        # Print loss and consuming time every epoch
        if (epoch + 1) % 1 == 0:
            # print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))
            # loss1 = np.append(loss1,loss.item())
            print('Epoch: {:d} finished ! Data Loss: {:.5f}'.format(epoch + 1, loss_data))
            print('Epoch: {:d} finished ! Model Loss: {:.5f}'.format(epoch + 1, loss_model))
            print('Epoch: {:d} finished ! SSIM Model Loss: {:.5f}'.format(epoch + 1, loss_model_ssim))
            time_elapsed = time.time() - since
            print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

v_save.tofile('fin_lbfgs_marmousi2_inv.bin')
print('lowest model loss is :',loss_model_count)
# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

data = {}
data['loss_model'] = loss_model_sum
scipy.io.savemat('ModelLoss.mat', data)

data = {}
data['loss_data'] = loss_data_sum
scipy.io.savemat('DataLoss.mat', data)

data = {}
data['loss_data'] = loss_model_ssim_sum
scipy.io.savemat('SSIMModelLoss.mat', data)

loss_data_sum = (loss_data_sum - np.min(loss_data_sum)) / (np.max(loss_data_sum) - np.min(loss_data_sum))
loss_model_sum = (loss_model_sum - np.min(loss_model_sum)) / (np.max(loss_model_sum) - np.min(loss_model_sum))

fig = plt.figure(figsize=(10.5, 3.5), dpi=300)
plt.plot(np.arange(len(loss_data_sum)), loss_data_sum, label='data loss')
plt.plot(np.arange(len(loss_model_sum)), loss_model_sum, label='model loss')
plt.plot(np.arange(len(loss_model_ssim_sum)), loss_model_ssim_sum, label='SSIM model loss')
plt.legend()
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
# plt.ylim(-0.05,1.05)
plt.title('Data and model loss', fontsize=15)
plt.savefig('data_model_loss.jpg', dpi=300, transparent=True, bbox_inches='tight')




