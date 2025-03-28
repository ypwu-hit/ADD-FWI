# @author: wuyuping (ypwu@stu.hit.edu.cn)

import time
import torch
import numpy as np
from data_loader import DataLoad_Train
from deep_decoder import VanillaConvUpDecoder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from deepwave import scalar
import scipy.io
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
import random
from scipy.ndimage import gaussian_filter
import math


# this Adam is introduced in the following reference.
# Self-supervised Deep Image Restoration via Adaptive Stochastic Gradient Langevin Dynamics
# https://github.com/Wang-weixi/Adaptive_Sampling/blob/main/utils/common_utils.py
class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, noise_level,params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        self.noise_level=noise_level
    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    
    def step(self, loss_data,noise_type='gaussian',closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                ssize=grad.size()
                #zzero=torch.zeros(ssize).cuda()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                #norm=torch.norm(grad)
                #print('grad_norm',norm)
                # noise = torch.cuda.FloatTensor(ssize).to(torch.device("cuda:1"))
                noise = torch.randn_like(grad)
                #torch.randn(ssize, out=noise)
                # if noise_type=='poission':
                #     torch.randn(ssize, out=noise)  
                #     possion=torch.poisson(grad)
                #     if torch.norm(possion)>1*torch.norm(grad):
                #         possion=possion-grad
                #         noise=10*possion
                # if noise_type=='gaussian':
                #     torch.randn(ssize, out=noise)
                # if noise_type=='bernoulli':
                #     torch.ones(ssize,out=noise)
                #     noise=0.1*noise
                #     torch.bernoulli(noise, out=noise)
                #     noise=-noise*grad
                # if noise_type=='levi':
                #     torch.randn(ssize, out=noise)  
                #     possion=torch.poisson(grad)
                #     if torch.norm(possion)>1*torch.norm(grad):
                #         possion=possion-grad
                #         noise=30*possion+noise
                            
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg=exp_avg+(1 - beta1)*loss_data*self.noise_level*noise
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return (super(SSIM_Loss, self).forward(img1, img2) )

class MyDataset(data_utils.Dataset):
    def __init__(self, seismic_data, source_amplitudes, source_locations, receiver_locations):

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

def get_parameter_number(net):

    # calculating the parameters of network

    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    return {'Total': total_num, 'Trainable': trainable_num}

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    
    

same_seeds(1234)

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

INPUT = 'noise'
BatchSize = 6
number_of_shots = 30
dx = 22.5
dt = 0.002
freq = 7

seismic_data, seismic_data_noise, data_dim, model_dim, source_amplitudes, x_s, x_r, v_true = DataLoad_Train(number_of_shots)
# print('source_amplitudes', source_amplitudes.shape, x_s.shape, x_r.shape)

v_true = v_true.to(device)

v_mig = torch.tensor(1 / gaussian_filter(1 / v_true.detach().cpu().numpy(), 30))
v_mig = v_mig.to(device)

seismic_data_size = seismic_data.size()

model_max = torch.max(v_true)
model_min = torch.min(v_true)
v_true_nor = (v_true - model_min) / (model_max - model_min)

net_input = torch.randn(1,64,11,5)
print('net_input.shape', net_input.shape)
net_input.numpy().tofile('net_input.bin')

train_loader = data_utils.DataLoader(MyDataset(seismic_data_noise, source_amplitudes, x_s, x_r),
                                     batch_size=BatchSize,
                                     shuffle=True, drop_last=False)

net = VanillaConvUpDecoder()
print(get_parameter_number(net))

torch.save(net.state_dict(),'nnfwi_decoder_init.pkl')

print(net)
net.to(device)

print()
print('*******************************************')
print('*******************************************')
print('           START TRAINING         ')
print('*******************************************')
print('*******************************************')
print()

v = v_mig.clone()
v.requires_grad_()


optimizer_v = Adam(5e-7,[v], lr=25)
# optimizer_v = torch.optim.Adam([v], lr=25)
optimizer_w = torch.optim.Adam(net.parameters())

loss_fn = torch.nn.MSELoss()
loss_ssim = SSIM_Loss(data_range=1.0, size_average=True, channel=1)

DisplayStep = 2
loss_data_file = 'loss_data.txt'
loss_model_file = 'loss_model.txt'
loss_ssim_file = 'loss_ssim.txt'
step   = int(number_of_shots/BatchSize)
num_epochs = 1001
start  = time.time()

since = time.time()

v_cme = torch.zeros_like(v_mig)
count = 0.0

for epoch in range(num_epochs):

    net.train()

    for i, (s_d_i, s_a_i, x_s_i, x_r_i) in enumerate(train_loader):

        optimizer_v.zero_grad()
        
        # Forward prediction
        outputs = net(net_input.to(device)).mean(dim=[0,1], keepdim=False) + v_mig
        outputs = torch.clamp(outputs, min=1100, max=4700)

        out = scalar(
            v.to(device), dx, dt,
            source_amplitudes=s_a_i.to(device),
            source_locations=x_s_i.to(device),
            receiver_locations=x_r_i.to(device),
            max_vel=4700,
            pml_freq=freq,
            accuracy=8,
            pml_width=[20, 20, 0, 20]
        )[-1]
        loss_1 = loss_fn(out,s_d_i.to(device))
        loss_2 = loss_1.item()*1e-4*loss_fn(v, outputs.detach())
        loss_v = loss_1 + loss_2

        if np.isnan(float(loss_v.item())):
            raise ValueError('loss is nan while training')

        # Loss backward propagation
        loss_v.backward()
        optimizer_v.step(1)
        v.data = torch.clamp(v.data, min=1100, max=4700)
        # print(loss_1.item(),loss_2.item())

        if epoch >= 500:
            v_cme = v_cme + v.detach()
            count += 1

        for index in range(10):

            optimizer_w.zero_grad()
            loss_w = loss_fn(v.detach(), outputs)
            loss_w.backward()
            optimizer_w.step()

            outputs = net(net_input.to(device)).mean(dim=[0,1], keepdim=False) + v_mig
            outputs = torch.clamp(outputs, min=1100, max=4700)


    if epoch % 20 == 0:

        # val
        with torch.no_grad():  # operations inside don't track history
            # Validation Mode:

            observed_data_fd = scalar(v.to(device), dx, dt,
                                      source_amplitudes=source_amplitudes.to(device),
                                      source_locations=x_s.to(device),
                                      receiver_locations=x_r.to(device),
                                      max_vel=4700,
                                      pml_freq=freq,
                                      accuracy=8,
                                      pml_width=[20, 20, 0, 20])[-1]

            loss_model = loss_fn(v, v_true) / loss_fn(v_true, torch.zeros_like(v_true))

            if epoch == 0:
                loss_model_count = loss_model.detach().cpu()
                v_save = v.detach().cpu().numpy()
            else:
                if loss_model.detach().cpu() < loss_model_count:
                    loss_model_count = loss_model.detach().cpu()
                    v_save = v.detach().cpu().numpy()


            loss_data = loss_fn(observed_data_fd, seismic_data.to(device)) / loss_fn(seismic_data.to(device), 
                torch.zeros_like(seismic_data.to(device)))

            model_max = torch.max(v)
            model_min = torch.min(v)
            model_nor = (v - model_min) / (model_max - model_min)
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

            if epoch >= 500:
                v_cme_vis = v_cme/count

                plt.figure()
                plt.imshow(v_cme_vis.detach().cpu().numpy().T, aspect='auto', cmap='jet', vmax=4700, vmin=1100)
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.title("Velocity")
                plt.colorbar()

                plt.tight_layout()
                plt.savefig(str(epoch) + '_v_cme_vis.png')
                plt.close()

                v_cme_vis.detach().cpu().numpy().tofile(str(epoch) + '_v_cme_vis.bin')

            
            plt.figure()
            plt.imshow(v.detach().cpu().numpy().T, aspect='auto', cmap='jet', vmax=4700, vmin=1100)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title("Velocity")
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(str(epoch) + '.png')
            plt.close()

            v.detach().cpu().numpy().tofile(str(epoch) + 'add_wdp_marmousi2_inv.bin')

            # Validation Mode:
            net.eval()

            outputs = net(net_input.to(device)).mean(dim=[0,1], keepdim=False) + v_mig
            outputs = torch.clamp(outputs, min=1100, max=4700)
            plt.figure()
            plt.imshow(outputs.detach().cpu().numpy().T, aspect='auto', cmap='jet', vmax=4700, vmin=1100)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title("Velocity")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(str(epoch) + 'add_wdp.png')
            plt.close()

            outputs.detach().cpu().numpy().tofile(str(epoch) + '_add_wdp_marmousi2_inv.bin')
            torch.save(net.state_dict(),'nnfwi3_add.pkl')

            # Print loss and consuming time every epoch
            # print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))
            # loss1 = np.append(loss1,loss.item())
            print('Epoch: {:d} finished ! Data Loss: {:.5f}'.format(epoch + 1, loss_data))
            print('Epoch: {:d} finished ! Model Loss: {:.5f}'.format(epoch + 1, loss_model))
            print('Epoch: {:d} finished ! SSIM Model Loss: {:.5f}'.format(epoch + 1, loss_model_ssim))
            time_elapsed = time.time() - since
            print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            since = time.time()

v_cme /= count
v_cme.detach().cpu().numpy().tofile('v_cme.bin')

v_save.tofile('fin_unet_marmousi2_inv.bin')
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



