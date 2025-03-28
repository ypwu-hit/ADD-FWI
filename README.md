ADD-FWI
---------
This repo contains a PyTorch implementation with [Deepwave](https://ausargeo.com/deepwave/) for the paper How does Neural Network Reparametrization Improve Geophysical Inversion? which is published on the Journal of Geophysical Research: Machine Learning and Computation.

Abstract
---------
Full waveform inversion (FWI) is a high-resolution seismic inversion technique and great efforts have been made to mitigate the multi-solution problem, such as the traditional total variation (TV) regularization.
Different from traditional regularization, a new regularization design approach named neural network reparametrization (\emph{Deep Image Prior}) was presented recently. 
The existing neural network parametrization-based FWI (NN-FWI) was implemented using an over-parametrization framework.
On the contrary, we adopt an under-parameterized framework \emph{Deep Decoder} (DD) and propose a new under-parameterized neural network regularization framework, so-called \emph{Attention Deep Decoder} (ADD). 
Further applying DD and ADD to seismic inversion, we propose DD-FWI and ADD-FWI, a new formulation of NN-FWI that uses an under-parameterised network to represent the velocity model in FWI and minimizes an objective function over the network parameters.
Inspired from this formulation, NN reparametrization can be a model-domain multiscale strategy and the interpolation operator is the key component to regularize the inversion.
Besides, we discover the potential relationship between interpolation-based reparametrization, traditional TV regularization and wavelet transform from the mathematical aspect.
Experiments show the effectiveness of our proposal overcoming the requirement of initial model in the case of data obtained from a streamer acquisition system and lacking the low-frequency component below 10Hz.
Moreover, the comparison experiments of FWI using TV regularization, over- and under-parameterized NN regularization indicate that the proposed method might move further towards practical application and proves a way to develop inverse problems through an under-parameterized neural network regularization framework.
