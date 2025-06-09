import numpy as np
import torch
import taichi as ti
from FEM.Sh_parameter import params


@ti.data_oriented
class Sh_material_initial:
    def __init__(self):

        s_maxNumnp = params.s_maxNumnp[None]
        s_maxGP = params.s_maxGP[None]
        s_maxMat = params.s_maxMat[None]

        self.s_MatType = ti.field(ti.i32, shape=(s_maxNumnp))  #!material type of particle
        self.s_MatTypek = ti.field(ti.i32, shape=(s_maxGP))    #!material type of gauss pt.
        self.ipl_model = ti.field(ti.i32, shape=(s_maxGP))     #!hardening type of material
        
        # elastic properties
        self.el_rho = ti.field(ti.f32, shape=(s_maxMat))
        self.el_E = ti.field(ti.f32, shape=(s_maxMat))
        self.el_V = ti.field(ti.f32, shape=(s_maxMat))
        self.el_G = ti.field(ti.f32, shape=(s_maxMat))
        self.el_K = ti.field(ti.f32, shape=(s_maxMat))
        

        # Plastic properties (linear hardening)
        self.pl_k0 = ti.field(ti.f32, shape=(s_maxMat))
        self.pl_EP = ti.field(ti.f32, shape=(s_maxMat))
        # Plastic properties (exponential hardening)
        self.pl_a = ti.field(ti.f32, shape=(s_maxMat))
        self.pl_e0 = ti.field(ti.f32, shape=(s_maxMat))
        self.pl_n = ti.field(ti.f32, shape=(s_maxMat))
        # Plastic properties (Johnson-Cook model)
        self.JC_A = ti.field(ti.f32, shape=(s_maxMat))
        self.JC_B = ti.field(ti.f32, shape=(s_maxMat))
        self.JC_n = ti.field(ti.f32, shape=(s_maxMat))
        self.JC_C = ti.field(ti.f32, shape=(s_maxMat))
        self.JC_m = ti.field(ti.f32, shape=(s_maxMat))
        # Strain rate effect
        self.cs_D = ti.field(ti.f32, shape=(s_maxMat))
        self.cs_n = ti.field(ti.f32, shape=(s_maxMat))
        

        # Damage properties (Lam damage)
        self.lm_D_c = ti.field(ti.f32, shape=(s_maxMat))
        self.lm_stn_pr = ti.field(ti.f32, shape=(s_maxMat))
        self.lm_stn_pd = ti.field(ti.f32, shape=(s_maxMat))
        # Damage properties (GTN damage)
        self.q1 = ti.field(ti.f32, shape=(s_maxMat))
        self.q2 = ti.field(ti.f32, shape=(s_maxMat))
        self.q3 = ti.field(ti.f32, shape=(s_maxMat))
        

        # Material property initialization for different steel types
        # Q235 Steel
        self.el_rho[0] = 7850 #2700
        self.el_E[0] = 2.0e11 #67.5e9
        self.el_V[0] = 0.3 #0.34
        self.el_G[0] = self.el_E[0] / (2 * (1 + self.el_V[0]))
        self.el_K[0] = self.el_E[0] / (3 * (1 - 2 * self.el_V[0]))
        # plastic
        self.pl_k0[0] = 2.35e18
        self.pl_EP[0] = 5.0e18
        self.pl_a[0] = 0.0
        self.pl_e0[0] = 0.0
        self.pl_n[0] = 0.0
        self.cs_D[0] = 40
        self.cs_n[0] = 5
        
        self.JC_A[0] = 249.2e6
        self.JC_B[0] = 889e6
        self.JC_n[0] = 0.746
        self.JC_C[0] = 0.058
        self.JC_m[0] = 0.94
        # damage
        self.lm_D_c[0] = 0.3
        self.lm_stn_pr[0] = 0.28
        self.lm_stn_pd[0] = 0.20
    
        # Q921 Steel
        self.el_rho[1] = 7850
        self.el_E[1] = 2.11e11
        self.el_V[1] = 0.3
        self.el_G[1] = self.el_E[1] / (2 * (1 + self.el_V[1]))
        self.el_K[1] = self.el_E[1] / (3 * (1 - 2 * self.el_V[1]))
        # plastic
        self.pl_k0[1] = 7.14e18
        self.pl_EP[1] = 2.0e9
        self.pl_a[1] = 0.0
        self.pl_e0[1] = 0.0
        self.pl_n[1] = 0.0
        self.cs_D[1] = 6.0e-4
        self.cs_n[1] = 50.8
        # damage
        self.lm_D_c[1] = 0.40
        self.lm_stn_pr[1] = 0.28
        self.lm_stn_pd[1] = 0.20
        

        # HSLA Steel
        self.el_rho[2] = 7850
        self.el_E[2] = 2.11e11
        self.el_V[2] = 0.3
        self.el_G[2] = self.el_E[2] / (2 * (1 + self.el_V[2]))
        self.el_K[2] = self.el_E[2] / (3 * (1 - 2 * self.el_V[2]))
        # plastic
        self.pl_k0[2] = 4.00e8
        self.pl_EP[2] = 1.305e9
        self.pl_a[2] = 720.0e6
        self.pl_e0[2] = 0.025
        self.pl_n[2] = 0.16
        self.cs_D[2] = 40
        self.cs_n[2] = 5
        # damage
        self.lm_D_c[2] = 0.32
        self.lm_stn_pr[2] = 0.28
        self.lm_stn_pd[2] = 0.20
    
# 获取全局实例
mat = Sh_material_initial()