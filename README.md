ADD-FWI
---------
This repo contains a PyTorch implementation with [Deepwave](https://ausargeo.com/deepwave/) for the paper How does Neural Network Reparametrization Improve Geophysical Inversion? which is published on the Journal of Geophysical Research: Machine Learning and Computation.


![](/images/Figure1.svg)


Abstract
---------
Full waveform inversion (FWI) is a high-resolution seismic inversion technique and great efforts have been made to mitigate the multi-solution problem, such as the traditional total variation (TV) regularization. Different from traditional regularization, a new regularization design approach named neural network reparametrization (*Deep Image Prior*) was presented recently. The existing neural network parametrization-based FWI (NN-FWI) was implemented using an over-parametrization framework.
On the contrary, we adopt an under-parameterized framework *Deep Decoder* (DD) and propose a new under-parameterized neural network regularization framework, so-called *Attention Deep Decoder* (ADD). Further applying DD and ADD to seismic inversion, we propose DD-FWI and ADD-FWI, a new formulation of NN-FWI that uses an under-parameterised network to represent the velocity model in FWI and minimizes an objective function over the network parameters. Inspired from this formulation, NN reparametrization can be a model-domain multiscale strategy and the interpolation operator is the key component to regularize the inversion. Besides, we discover the potential relationship between interpolation-based reparametrization, traditional TV regularization and wavelet transform from the mathematical aspect. Experiments show the effectiveness of our proposal overcoming the requirement of initial model in the case of data obtained from a streamer acquisition system and lacking the low-frequency component below 10Hz. Moreover, the comparison experiments of FWI using TV regularization, over- and under-parameterized NN regularization indicate that the proposed method might move further towards practical application and proves a way to develop inverse problems through an under-parameterized neural network regularization framework.

Plain Language Summary
---------
Seismic inversion solved by FWI is a nonlinear and ill-posed inverse problem, which suffers from local minima and multiple solution issues. Great efforts have been made, such as multiscale inversion strategy, misfit function modification and regularization techniques. A new regularization formulation for FWI implemented by an over-parametrization deep neural network (DNN) framework was proposed, which couples advanced DNN techniques to impose regularization effect into the inversion process. However, it is not clear enough what components play a key regularizing role for inversion and provide what kind of regularization effects. We propose an under-parametrization regularization framework for FWI, realizing that this reparametrization can be seen as a model-domain multiscale strategy and the interpolation operator offers a key regularization effect. Therefore, this framework updates the model (e.g., velocity) using the gradient with frequency bias and regularizes the update with proximity similarity, which mitigates the cycle skipping and constraints the solution space.

Prerequisites
---------
```
python 3.10.13  
torch 2.1.1+cu121
torchaudio 2.1.1+cu121
torchvision 0.16.1+cu121
scipy 1.11.4
numpy 1.24.1
matplotlib 3.8.2
scikit-image 0.25.0
einops 0.7.0
deepwave 0.0.20
pytorch-msssim 1.0.0
```

Run the code
-----------
Enter the ADD-FWI folder
```
cd ./ADD-FWI/sesimic-data-generate/
```
Correct the parameter settings and data path in ```data_generate.ipynb``` and ```data_generate_streamer_acquisition.ipynb```
e.g.
```
####################################################
####                   FILENAMES               ####
####################################################
data_path = ''
ny = 340
nx = 130
v_true = torch.from_file(data_path+'mar_big_vp_130_340.bin',
                    size=ny * nx).reshape(ny, nx)
```
and
```
####################################################
####   MAIN PARAMETERS FOR FORWARD MODELING         ####
####################################################
dx = 22.5                # step interval along x/z direction
n_shots = 30             # total number of source

n_sources_per_shot = 1   # number of sources per shot 
d_source = 11            # step interval of source
first_source = 5         # the first position of source
source_depth = 1         # the depth position of source

n_receivers_per_shot = 339 # number of receiver 
d_receiver = 1           # step interval of receiver
first_receiver = 0       # the first position of receiver
receiver_depth = 1       # the depth position of receiver

freq = 7                 # central frequency
nt = 2000
dt = 0.002               # time interval (e.g., 2ms)
peak_time = 1.0 / freq   # the time (in secs) of the peak amplitude
```
Then, run the following script to generate dataset and implement FWIs, NN-FWIs, WDP-FWIs, etc. 
```data_generate.ipynb``` and ```data_generate_streamer_acquisition.ipynb```
The initla model, source amplitude, and observed data are saved in the result path. Next, run the following script to implement FWI, NN-FWIs, WDP-FWIs, etc.
```
cd ./ADD-FWI/NN-FWIs/ADD-FWI/
```
```
python train.py
```

Results
----------
Outputs can be found in ```/ADD-FWI/images/...```.
### Visual examples
#### 1. Marmousi2 Model:
![Inversion results of DD-FWI.](/images/Figure2.svg)

#### 2. Marmousi2 Model:
![Inversion results of NN-FWIs.](/images/Figure3.svg)



If you have any questions about this work, feel free to contract us: ypwu@stu.hit.edu.cn
