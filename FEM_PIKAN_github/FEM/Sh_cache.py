import numpy as np
import torch
import taichi as ti
from FEM.Sh_parameter import params


@ti.data_oriented
class Sh_variable_allocate:
    def __init__(self):

        s_ndim = params.s_ndim[None]
        s_maxNumnp = params.s_maxNumnp[None]
        s_maxElem = params.s_maxElem[None]
        s_maxGP = params.s_maxGP[None]
        s_maxntg = params.s_maxntg[None]
        s_maxNode = params.s_maxNode[None]
        s_maxDispbcX = params.s_maxDispbcX[None]
        s_maxDispbcY = params.s_maxDispbcY[None]
        s_maxDispbcZ = params.s_maxDispbcZ[None]
        s_maxVelbcX = params.s_maxVelbcX[None]
        s_maxVelbcY = params.s_maxVelbcY[None]
        s_maxVelbcZ = params.s_maxVelbcZ[None]
        s_maxAngDispbcX = params.s_maxAngDispbcX[None]
        s_maxAngDispbcY = params.s_maxAngDispbcY[None]
        s_maxAngDispbcZ = params.s_maxAngDispbcZ[None]
        s_maxAngVelbcX = params.s_maxAngVelbcX[None]
        s_maxAngVelbcY = params.s_maxAngVelbcY[None]
        s_maxAngVelbcZ = params.s_maxAngVelbcZ[None]
        s_maxEssbcX = params.s_maxEssbcX[None]
        s_maxEssbcY = params.s_maxEssbcY[None]
        s_maxEssbcZ = params.s_maxEssbcZ[None]
        s_maxEssbc = params.s_maxEssbc[None]
        s_maxAngEssbcX = params.s_maxAngEssbcX[None]
        s_max_contact_node = params.s_max_contact_node[None]
        s_maxNtip = params.s_maxNtip[None]
        s_mnsch = params.s_mnsch[None]
        s_maxCrack = params.s_maxCrack[None]
        s_contact = params.s_contact[None]
        s_maxInt = params.s_maxInt[None]
        s_maxIntElem = params.s_maxIntElem[None]
        
        self.d = ti.field(ti.i32, 24)
        self.s_Kvv = ti.Matrix.field(3, 3, ti.f32, shape=(4, 4, s_maxGP), needs_grad=False)
        self.s_Kvw = ti.Matrix.field(3, 3, ti.f32, shape=(4, 4, s_maxGP), needs_grad=False)
        self.s_Kwv = ti.Matrix.field(3, 3, ti.f32, shape=(4, 4, s_maxGP), needs_grad=False)
        self.s_Kww = ti.Matrix.field(3, 3, ti.f32, shape=(4, 4, s_maxGP), needs_grad=False)
        self.Bv = ti.Matrix.field(6, 3, ti.f32, shape=(4, s_maxGP, s_maxntg))
        self.Bw = ti.Matrix.field(6, 3, ti.f32, shape=(4, s_maxGP, s_maxntg))
        self.Bv_T = ti.Matrix.field(3, 6, ti.f32, shape=(4, s_maxGP, s_maxntg))
        self.Bw_T = ti.Matrix.field(3, 6, ti.f32, shape=(4, s_maxGP, s_maxntg))
        self.nonzero = ti.field(ti.i32, shape=()) 
        self.random = ti.field(ti.f32, shape=(s_maxNumnp))
        self.Disp = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp), needs_grad=True)
        self.Angdisp = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp), needs_grad=True)

        self.disp_FEM = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp))
        self.angdisp_FEM = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp))
        self.disp_error = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp))
        self.angdisp_error = ti.Vector.field(3, ti.f32, shape=(s_maxNumnp))
        self.disp_FEM_mean = ti.field(ti.f32, shape=())
        self.angdisp_FEM_mean = ti.field(ti.f32, shape=())
        self.disp_MSE = ti.field(ti.f32, shape=())
        self.angdisp_MSE = ti.field(ti.f32, shape=())
        self.total_MSE = ti.field(ti.f32, shape=())
        self.disp_RMSE = ti.field(ti.f32, shape=())
        self.angdisp_RMSE = ti.field(ti.f32, shape=())
        self.total_RMSE = ti.field(ti.f32, shape=())

        self.s_nfree = ti.field(ti.i32, shape=())
        self.s_ntotal = ti.field(ti.i32, shape=())       #!total num of node and gauss pt.
        self.s_nelement = ti.field(ti.i32, shape=())     #!num of shell element
        self.s_nnode = ti.field(ti.i32, shape=())        #!num of shell node
        self.s_mgk = ti.field(ti.i32, shape=())          #!num of gauss pt.
        self.s_nintElem = ti.field(ti.i32, shape=())     #!gauss pt. of each element
        self.s_ele_node = ti.field(ti.i32, shape=())     #!node of each element
        self.s_xm = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))        # Coordinate of particle
        self.s_xs = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))        # Smoothed coordinate of particle
        self.s_xm0 = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))       # Initial coordinate of particle
        

        self.s_nele_node = ti.field(ti.i32, shape=(5, s_maxElem))   #!单元，节点1，节点2，...
        self.s_gp_ele = ti.field(ti.i32, shape=(2, s_maxGP))        #!Gauss积分点对应的单元编号,单元编号,积分点顺序编号(对于多个积分点的情形)
        self.s_ele_gp = ti.field(ti.i32, shape=(s_maxntg + 1, s_maxElem))  #!单元对应的gauss积分点编号,单元内积分点个数,积分点1,积分点2
        self.s_dvom =  ti.field(ti.f32, shape=(s_maxNumnp))                # Volume of particle
        self.s_darea =  ti.field(ti.f32, shape=(s_maxNumnp))               # Area of particle
        self.s_dvomk =  ti.field(ti.f32, shape=(s_maxGP))                  # Volume of Gauss pt.
        self.s_dareak =  ti.field(ti.f32, shape=(s_maxGP))                 # Area of Gauss pt.
        self.s_thk =  ti.field(ti.f32, shape=(s_maxNumnp))                 # Thickness of shell at node
        
        self.s_mass_1 =  ti.field(ti.f32, shape=(s_maxNumnp))              # 1/mass of particle
        self.s_inertia_1 =  ti.field(ti.f32, shape=(s_maxNumnp))           # 1/inertia of particle
        self.s_tom = ti.field(ti.f32, shape=())                                # Total mass of the model
        self.dvtemp = ti.field(ti.f32, shape=()) 
        self.s_mass =  ti.field(ti.f32, shape=(s_maxNumnp))
        self.s_inertia =  ti.field(ti.f32, shape=(s_maxNumnp))
        
        
        self.s_disp = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp), needs_grad=False)      # Displacement of each node
        self.s_vel = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))       # Velocity of each node
        self.s_acc = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))       # Acceleration of each node
        self.s_angvel = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))    # Angular velocity of each node
        self.s_angdisp = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp), needs_grad=False)   # Angular displacement of each node
        self.s_angacc = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))    # Angular acceleration of each node
        

        self.s_xmk = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP))      #!coordinate of gauss pt.
        self.s_xmk0 = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP))     #!initial coordinate of gauss pt.
        self.s_thkk =  ti.field(ti.f32, shape=(s_maxGP))                  #!thickness of shell at gauss pt.
        self.s_dispk = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP))    #!displacement of gauss pt.
        self.s_velk = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP))     #!velocity of guass pt.
        self.s_angdispk = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP)) #!angular displacement of gauss pt.
        self.s_angvelk = ti.Vector.field(s_ndim,ti.f32, shape=(s_maxGP))  #!angular velocity of gauss pt.


        self.s_stsgp = ti.field(ti.f32, shape=(6, s_maxntg, s_maxGP))       # stress product of gauss pt.(s_maxntg is int point along thickness)
        self.s_effstsgp = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))       # effective stress of gauss pt.
        self.s_effstegp = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))       # effective strain of gauss pt.
        self.s_effstsnp =  ti.field(ti.f32, shape=(s_maxNumnp))
        self.s_effEopt =  ti.field(ti.f32, shape=(s_maxNumnp))                 # sum of shape function(≈1); Fill effEopt for weight average
        self.s_back_stress_gp = ti.field(ti.f32, shape=(5, s_maxntg, s_maxGP))  # back stress of gauss pt.(s_maxntg is int point along thickness)
        self.s_plastic_strain_gp = ti.field(ti.f32, shape=(5, s_maxntg, s_maxGP))  # plastic strain of gauss pt.(s_maxntg is int point along thickness)
        

        self.s_yield_sts_gp = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))   # yield stress of gauss pt.
        self.s_thermo = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))         # temperature of gauss pt.
        self.s_effRgp = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))         # 
        

        self.s_node_node = ti.field(ti.i32, shape=(25, s_maxNumnp))
        # !node_node:节点编号,公用节点数,逆时针第一个节点,第二个节点,...
        # !如果全部都采用结构化网格,没有加筋时,abaqus里一个节点最多属于5个单元,如果有某两个相邻单元不共边则最多会有6个节点,程序里最多只考虑了5个点的情况。
        # !有加筋时,最多允许周围有6个节点
        self.s_node_ele = ti.field(ti.i32, shape=(25, s_maxNumnp))   # node_ele: 节点编号,公用单元数,单元1,单元2,...  在有加筋的情况下最多一个节点允许有8个单元
        
        # Basic vectors and transformation matrices
        self.s_e_la1 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxGP))           # first basic vector of lamina coordinate systems at gauss pt.
        self.s_e_la2 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxGP))           # second basic vector of lamina coordinate systems at gauss pts
        self.s_e_la3 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxGP))           # third basic vector of lamina coordinate systems at gauss pt.
        self.s_q_gl = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxGP))    # transform matrix: from global to lamina
        self.s_q_gl0 = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxGP))   # initial transform matrix: from global to lamina
        
        self.s_e_la_n1 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))      # nodal normal of lamina
        self.s_e_la_n2 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))     # nodal normal of lamina
        self.s_e_la_n3 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))     # nodal normal of lamina
        self.s_e_fi1 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))       # first basic vector of fiber coordinat system at node
        self.s_e_fi2 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))       # second basic vector of fiber coordinat system at node
        self.s_e_fi3 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))       # third basic vector of fiber coordinat system at node
        self.s_e_fi1_t0 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))    # save the initial fiber coordinate basis
        self.s_e_fi2_t0 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))    # save the initial fiber coordinate basis
        self.s_e_fi3_t0 = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))    # save the initial fiber coordinate basis
        self.s_q_fg = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxNumnp)) # transform matrix: from fiber to global
        self.s_q_fg0 = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxNumnp)) # initial transform matrix: from fiber to global
        

        self.s_shpk = ti.field(ti.f32, shape=(s_maxNode, s_maxGP))         # shape function value of gauss pt. (g-n)
        self.s_shpkdxl = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.s_shpkdyl = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.s_shpkdzl = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.s_shpkdxl_hat = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.s_shpkdyl_hat = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.s_shpkdzl_hat = ti.field(ti.f32, shape=(s_maxNode, s_maxGP, s_maxntg))
        self.Jacobil = ti.Matrix.field(2, 2, ti.f32, shape=(s_maxGP))
        self.JacobiInvl = ti.Matrix.field(2, 2, ti.f32, shape=(s_maxGP))


        self.s_ibtype =  ti.field(ti.i32, shape=(s_maxNumnp))      # c...用于指示裂纹边缘节点粒子(=300),内部粒子(=0),自由边界(=100),本质边界(=200)
        self.s_ibtypek =  ti.field(ti.i32, shape=(s_maxGP))
        self.s_nDispbcX = ti.field(ti.i32, shape=())
        self.s_nDispbcY = ti.field(ti.i32, shape=())
        self.s_nDispbcZ = ti.field(ti.i32, shape=())
        self.s_LmDispbcX =  ti.field(ti.f32, shape=(s_maxDispbcX))
        self.s_LmDispbcY =  ti.field(ti.f32, shape=(s_maxDispbcY))
        self.s_LmDispbcZ =  ti.field(ti.f32, shape=(s_maxDispbcZ))
        self.s_vDispbcX =  ti.field(ti.f32, shape=(s_maxDispbcX))
        self.s_vDispbcY =  ti.field(ti.f32, shape=(s_maxDispbcY))
        self.s_vDispbcZ =  ti.field(ti.f32, shape=(s_maxDispbcZ))
        

        self.s_nVelbcX = ti.field(ti.i32, shape=())
        self.s_nVelbcY = ti.field(ti.i32, shape=())
        self.s_nVelbcZ = ti.field(ti.i32, shape=())
        self.s_LmVelbcX =  ti.field(ti.f32, shape=(s_maxVelbcX))
        self.s_LmVelbcY =  ti.field(ti.f32, shape=(s_maxVelbcY))
        self.s_LmVelbcZ =  ti.field(ti.f32, shape=(s_maxVelbcZ))
        self.s_vVelbcX =  ti.field(ti.f32, shape=(s_maxVelbcX))
        self.s_vVelbcY =  ti.field(ti.f32, shape=(s_maxVelbcY))
        self.s_vVelbcZ =  ti.field(ti.f32, shape=(s_maxVelbcZ))
        

        self.s_nAngDispbcX = ti.field(ti.i32, shape=())
        self.s_nAngDispbcY = ti.field(ti.i32, shape=())
        self.s_nAngDispbcZ = ti.field(ti.i32, shape=())
        self.s_LmAngDispbcX =  ti.field(ti.f32, shape=(s_maxAngDispbcX))
        self.s_LmAngDispbcY =  ti.field(ti.f32, shape=(s_maxAngDispbcY))
        self.s_LmAngDispbcZ =  ti.field(ti.f32, shape=(s_maxAngDispbcZ))
        self.s_vAngDispbcX =  ti.field(ti.f32, shape=(s_maxAngDispbcX))
        self.s_vAngDispbcY =  ti.field(ti.f32, shape=(s_maxAngDispbcY))
        self.s_vAngDispbcZ =  ti.field(ti.f32, shape=(s_maxAngDispbcZ))
        

        self.s_nAngVelbcX = ti.field(ti.i32, shape=())
        self.s_nAngVelbcY = ti.field(ti.i32, shape=())
        self.s_nAngVelbcZ = ti.field(ti.i32, shape=())
        self.s_LmAngVelbcX =  ti.field(ti.f32, shape=(s_maxAngVelbcX))
        self.s_LmAngVelbcY =  ti.field(ti.f32, shape=(s_maxAngVelbcY))
        self.s_LmAngVelbcZ =  ti.field(ti.f32, shape=(s_maxAngVelbcZ))
        self.s_vAngVelbcX =  ti.field(ti.f32, shape=(s_maxAngVelbcX))
        self.s_vAngVelbcY =  ti.field(ti.f32, shape=(s_maxAngVelbcY))
        self.s_vAngVelbcZ =  ti.field(ti.f32, shape=(s_maxAngVelbcZ))
        

        self.s_vEssbcX_disp =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_disp =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_disp =  ti.field(ti.f32, shape=(s_maxEssbcZ))
        self.s_vEssbcX_vel =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_vel =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_vel =  ti.field(ti.f32, shape=(s_maxEssbcZ))
        self.s_vEssbcX_acc =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_acc =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_acc =  ti.field(ti.f32, shape=(s_maxEssbcZ))
        self.s_vEssbcX_angdisp =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_angdisp =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_angdisp =  ti.field(ti.f32, shape=(s_maxEssbcZ))
        self.s_vEssbcX_angvel =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_angvel =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_angvel =  ti.field(ti.f32, shape=(s_maxEssbcZ))
        self.s_vEssbcX_angacc =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_vEssbcY_angacc =  ti.field(ti.f32, shape=(s_maxEssbcY))
        self.s_vEssbcZ_angacc =  ti.field(ti.f32, shape=(s_maxEssbcZ))


        self.fbc_disp =  ti.field(ti.f32, shape=(s_maxEssbc))
        self.fbc_vel =  ti.field(ti.f32, shape=(s_maxEssbc))
        self.fbc_acc =  ti.field(ti.f32, shape=(s_maxEssbc))
        self.fbc_angdisp =  ti.field(ti.f32, shape=(s_maxEssbc))
        self.fbc_angvel =  ti.field(ti.f32, shape=(s_maxEssbc))
        self.fbc_angacc =  ti.field(ti.f32, shape=(s_maxEssbc))


        self.s_nEssbcX = ti.field(ti.i32, shape=())
        self.s_nEssbcY = ti.field(ti.i32, shape=())
        self.s_nEssbcZ = ti.field(ti.i32, shape=())
        self.s_LmEssbcX =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_LmEssbcY =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_LmEssbcZ =  ti.field(ti.f32, shape=(s_maxEssbcX))
        self.s_nAngEssbcX = ti.field(ti.i32, shape=())
        self.s_nAngEssbcY = ti.field(ti.i32, shape=())
        self.s_nAngEssbcZ = ti.field(ti.i32, shape=())
        self.s_LmAngEssbcX =  ti.field(ti.f32, shape=(s_maxAngEssbcX))
        self.s_LmAngEssbcY =  ti.field(ti.f32, shape=(s_maxAngEssbcX))
        self.s_LmAngEssbcZ =  ti.field(ti.f32, shape=(s_maxAngEssbcX))


        self.s_disp_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))     #!nodal displacement at last time step
        self.s_vel_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))      #!nodal velocity at last time step
        self.s_acc_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))      #!nodal acceleration at last time step
        self.s_angdisp_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))  #!nodal angular displacement at last time step
        self.s_angvel_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))   #!nodal angular velocity at last time step
        self.s_angacc_last = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))   #!nodal angular acceleration at last time step
        self.s_ddisp = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))         #!nodal displacement increment during single step
        self.s_dangdisp = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))      #!nodal angular displacement increment during single step


        self.Fs_body = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Ms_body = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Fs_press = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Ms_press = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Fs_edge = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Ms_edge = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Fs_hourg = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.Ms_hourg = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.s_hedge_force = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.s_hedge_moment = ti.Vector.field(s_ndim, ti.f32, shape=(s_maxNumnp))
        self.s_uf_body = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxGP))      #!unit mass body force(≈9.8)
        self.s_pressure =  ti.field(ti.f32, shape=(s_maxGP))               #!pressure value at gauss pt.
        self.Fsext = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp), needs_grad=False)      #!nodal external force vector
        self.Msext = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp), needs_grad=False)      #!nodal external moment vector


        self.Fsint = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp), needs_grad=True)       #!nodal internal force vector 
        self.Msint = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp), needs_grad=True)       #!nodal internal moment vector 
        self.s_delta_n = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))   #!fiber displacement at node
        self.s_delta_nk = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxGP))      #!fiber displacement at node 
        self.s_ddelta_n = ti.Vector.field(s_ndim ,ti.f32, shape=(s_maxNumnp))  #!increment of fiber displacement at node 


        self.s_gp_update_support =  ti.field(ti.f32, shape=(s_maxGP))
        self.s_complete_damagek =  ti.field(ti.f32, shape=(s_maxGP))
        self.s_complete_damage =  ti.field(ti.f32, shape=(s_maxNumnp))
        self.s_damagek =  ti.field(ti.f32, shape=(s_maxGP))
        self.s_D_m = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))     #!main part of strain rate
        self.s_D_eq = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))    #!effective part of strain rate
        self.s_void = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))    #!volume fraction of particle
        self.s_voidk = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))   #!volume fraction of gauss pt.
        self.s_damage =  ti.field(ti.f32, shape=(s_maxNumnp))           #!nodal damage value(<=1)
        self.s_damagek =  ti.field(ti.f32, shape=(s_maxGP))             #!Gauss pt. damage value(<=1)
        self.s_damagek_sec = ti.field(ti.f32, shape=(s_maxntg, s_maxGP))  #!Gauss pt. damage value(<=1)

        self.s_crack_tip_num = ti.field(ti.i32, shape=())                      #!裂纹尖端个数
        self.s_crack_tip_list =  ti.field(ti.f32, shape=(s_maxNtip))   #!裂纹尖端的应力点编号
        self.s_crack_tip_block_num = ti.field(ti.i32, shape=())                #!裂纹尖端储存块
        self.s_crack_tip_block =  ti.field(ti.f32, shape=(s_maxGP))    #!储存裂纹尖端生成的次序


        self.s_crack_tip_list_new =  ti.field(ti.f32, shape=(s_maxNtip))
        self.s_dgp_niack =  ti.field(ti.f32, shape=(s_maxNtip))
        self.s_dgp_pair_ik = ti.field(ti.f32, shape=(s_mnsch, s_maxNtip))
        self.s_crack_behind_num =  ti.field(ti.f32, shape=(s_maxNtip))
        self.s_crack_pair_behind = ti.field(ti.f32, shape=(4, s_maxNtip))
        self.s_crack_front_num =  ti.field(ti.f32, shape=(s_maxNtip))
        self.s_crack_pair_front = ti.field(ti.f32, shape=(4, s_maxNtip))
        self.s_crack_path = ti.field(ti.f32, shape=(2, s_maxCrack))
        self.s_damage_niacn =  ti.field(ti.f32, shape=(s_maxNumnp))
        

        #!c==========================================c
        #!c       define the contant matrixs         c
        #!c==========================================c
        #!c
        #!c...只需要定义一次的矩阵，设置为全局变量，在最开始赋值
        self.co_alpha = ti.field(ti.f32, shape=())
        self.co_shear_k = ti.field(ti.f32, shape=())
        self.c11 = ti.field(ti.f32, shape=())
        self.c12 = ti.field(ti.f32, shape=())
        self.c21 = ti.field(ti.f32, shape=())
        self.c22 = ti.field(ti.f32, shape=())
        self.c33 = ti.field(ti.f32, shape=())
        self.c44 = ti.field(ti.f32, shape=())
        self.c55 = ti.field(ti.f32, shape=())

        self.matrix_P = ti.field(ti.f32, shape=(3, 3))
        self.matrix_C = ti.field(ti.f32, shape=(3, 3))
        self.matrix_A = ti.field(ti.f32, shape=(5, 5))
        self.matrix_Q = ti.field(ti.f32, shape=(3, 3))
        self.matrix_Q5 = ti.field(ti.f32, shape=(5, 5))
        self.lammda_p = ti.field(ti.f32, shape=(3, 3))
        self.lammda_c = ti.field(ti.f32, shape=(3, 3))
        self.m_omega = ti.field(ti.f32, shape=(5, 5))
        
        self.Xi_1P1 = ti.field(ti.f32, shape=())
        self.Xi_1P2 = ti.field(ti.f32, shape=())
        self.Xi_1P3 = ti.field(ti.f32, shape=())
        self.current_ts = ti.field(ti.i32, shape=())
        self.nstart = ti.field(ti.i32, shape=())
        self.nframe = ti.field(ti.i32, shape=())
        self.istep = ti.field(ti.i32, shape=())
        self.nstep = ti.field(ti.i32, shape=())
        self.time_scale = ti.field(ti.i32, shape=())
        
        self.time = ti.field(ti.f32, shape=())
        self.FinalTime = ti.field(ti.f32, shape=())
        self.time_out = ti.field(ti.f32, shape=())
        self.f_dt = ti.field(ti.f32, shape=())
        self.s_dt = ti.field(ti.f32, shape=())
    
        self.fsi_ifrInp = ti.field(ti.i32, shape=())
        self.fsi_ifrInp[None] = 301
        self.time = ti.field(ti.f32, shape=())
        self.loss_f1 = ti.field(ti.f32, shape=())
        self.loss_f2 = ti.field(ti.f32, shape=())
        self.loss_FEM = ti.field(ti.f32, shape=())    
        self.x_in = ti.field(ti.f32, shape=())

        self.gp_loc2D = ti.field(ti.f32, shape=((2,s_maxIntElem)))
        self.gp_weight2D = ti.field(ti.f32, shape=(s_maxIntElem))
        self.node_temp = ti.field(ti.i32, shape=(2, 5))

        self.s_PMt = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxGP))
        self.s_MMt = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=(s_maxGP))

        self.du_dx = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=())
        self.F_matx = ti.Matrix.field(s_ndim, s_ndim, ti.f32, shape=())

var = Sh_variable_allocate()