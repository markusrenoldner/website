---
title: 'Periodic poisson problem in FEniCSX'
excerpt: "Solving the poisson problem on a periodic mesh in FEniCSX <br/><img src='/images/500x300.png'>"
collection: portfolio
---

Find \\(u\in H^1_0(\Omega)\\), \\(\Omega\subset\mathbb{R}^n\\), such that for all \\(\phi\in\mathcal{D}(\Omega)\\) we have

$$\int_\Omega \nabla u\cdot\nabla \phi = \int_\Omega f \phi.$$


```
import fenicsx
```
