# Spatial dependency structure learning with copulas.
This repoitory contains code for the Censored Spatial Gaussian Copula from my paper: [Probabilistic Rainfall Downscaling: Joint Generalized Neural Models
with Censored Spatial Gaussian Copula](https://arxiv.org/abs/2308.09827).

## JGNM
The first part of that paper is the JGNM with code [here](https://github.com/Rilwan-Adewoyin/NeuralGLM). It is a density estimation model for future rainfall based on temporal and spatial climate model predictors. It outputs three parameters for each location, with outputs looking like this:

![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/obs_p_rho_mu.png)

Thes three parameters are used to get zero-gamma mixture densities for each location. The densities in question:

![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/densities_shapes_20y.png)
![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/ts_density.png)

## Censore Spatial Gaussian Copula
The second part is the spatial dependence with the Censored copula. 

Here are a few examples of samples you can generate with our method.

![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/slideshow_observed_rain_day_3787.png)
![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/slideshow%20day%203787.gif)

![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/slideshow_observed_rain_day_1467.png)
![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/slideshow%20day%201467.gif)

![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/sliedshow%20observed%20rain%20day%202076.png)
![](https://github.com/Huk-David/SpaDep_VCop/blob/4Paper/body/Figures/slideshow%20day%202076.gif)



