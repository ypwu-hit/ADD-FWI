ADD-FWI
---------
This repo contains a PyTorch implementation with [Deepwave](https://ausargeo.com/deepwave/) for the paper How does Neural Network Reparametrization Improve Geophysical Inversion? which is published on the Journal of Geophysical Research: Machine Learning and Computation.


![NN-FWI](/images/NN-FWI.svg) &nbsp;&nbsp;  <img src="/images/ADD-FWI.jpg" width="700" height="350">

Abstract
---------
Full waveform inversion (FWI) is a high-resolution seismic inversion technique and great efforts have been made to mitigate the multi-solution problem, such as the traditional total variation (TV) regularization.
Different from traditional regularization, a new regularization design approach named neural network reparametrization (*Deep Image Prior*) was presented recently. 
The existing neural network parametrization-based FWI (NN-FWI) was implemented using an over-parametrization framework.
On the contrary, we adopt an under-parameterized framework *Deep Decoder* (DD) and propose a new under-parameterized neural network regularization framework, so-called *Attention Deep Decoder* (ADD). 
Further applying DD and ADD to seismic inversion, we propose DD-FWI and ADD-FWI, a new formulation of NN-FWI that uses an under-parameterised network to represent the velocity model in FWI and minimizes an objective function over the network parameters.
Inspired from this formulation, NN reparametrization can be a model-domain multiscale strategy and the interpolation operator is the key component to regularize the inversion.
Besides, we discover the potential relationship between interpolation-based reparametrization, traditional TV regularization and wavelet transform from the mathematical aspect.
Experiments show the effectiveness of our proposal overcoming the requirement of initial model in the case of data obtained from a streamer acquisition system and lacking the low-frequency component below 10Hz.
Moreover, the comparison experiments of FWI using TV regularization, over- and under-parameterized NN regularization indicate that the proposed method might move further towards practical application and proves a way to develop inverse problems through an under-parameterized neural network regularization framework.

Plain Language Summary
---------
Seismic inversion solved by FWI is a nonlinear and ill-posed inverse problem, which suffers from local minima and multiple solution issues.
Great efforts have been made, such as multiscale inversion strategy, misfit function modification and regularization techniques.  
A new regularization formulation for FWI implemented by an over-parametrization deep neural network (DNN) framework was proposed, which couples advanced DNN techniques to impose regularization effect into the inversion process.
However, it is not clear enough what components play a key regularizing role for inversion and provide what kind of regularization effects.
We propose an under-parametrization regularization framework for FWI, realizing that this reparametrization can be seen as a model-domain multiscale strategy and the interpolation operator offers a key regularization effect.
Therefore, this framework updates the model (e.g., velocity) using the gradient with frequency bias and regularizes the update with proximity similarity, which mitigates the cycle skipping and constraints the solution space.
