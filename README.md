# Extended-Lagrangian-SINDy-xL-SINDy-

## Overview
Extended Lagrangian SINDy (xL-SINDy) is an algorithm designed to obtain Lagrangian function of nonlinear dynamical systems from noisy measurement data. The Lagrangian function is modelled as a linear combination of nonlinear candidate functions, and Euler-
Lagrange’s equation is used to formulate the objective cost function. The optimization of the learning process is done with proximal gradient method. 

![overview](/images/overview.png)

For more detail about the derivation and the problem formulation, please take a look at this paper (in the process of submission to RAL with ICRA option 2022).

[Sparse Identification of Lagrangian for Nonlinear Dynamical Systems via Proximal Gradient Method](https://drive.google.com/file/d/14FqbwIONE2wfZJqi2hgxNPylV5eYjNQv/view?usp=sharing)

## Examples
The effectiveness of xL-SINDy  is demonstrated against different noise levels in physical simulation with four dynamical systems: A single pendulum, a cart-pendulum, a double pendulum, and a spherical pendulum.

![systems](/images/systems.png)

<p align="center">
  <img width="460" height="300" src="https://github.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-/blob/main/images/systems.png?raw=true">
</p>

## Installation
* Clone this repository `git clone https://github.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-.git`
* Install the environment and dependencies `conda env create -f environment.yml`

## How to Use
