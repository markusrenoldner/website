---
title: 'Solving a periodic poisson problem in FEniCSX'
date: 2025-04-23
permalink: /posts/2025/04/fenicsx_poisson_periodic/
tags:
  - fenicsx
---



# Solving a periodic poisson problem in FEniCSX

Find \\(u\in H^1_0(\Omega)\\), \\(\Omega\subset\mathbb{R}^n\\), such that for all \\(\phi\in\mathcal{D}(\Omega)\\) we have

$$\int_\Omega \nabla u\cdot\nabla \phi = \int_\Omega f \phi.$$


```
import fenicsx
```
