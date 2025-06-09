from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Gaussian import *
from FEM.Evl_fem_shape4 import *
import taichi as ti

@ti.kernel
def Sh_direct_find():
    # !c==========================================c
    # !c   Searching nodes for each gauss point   c
    # !c      already have array s_nele_node      c
    # !c==========================================c
    # !c
    print('*** Searching is done!')
    #!c==========================================c
    #!c **Eval and store Shape function NJ(xk):  c
    #!c    the shape functions at gaussion       c
    #!c     points in the support of any nodes   c
    #!c==========================================c
    #!c
    s_maxntg = params.s_maxntg[None]
    gp_loc1D, gp_weight1D = gauss1D(s_maxntg)
    var.s_shpk.fill(0.0)
    var.s_shpkdxl.fill(0.0)
    var.s_shpkdyl.fill(0.0)
    var.s_shpkdxl_hat.fill(0.0)
    var.s_shpkdyl_hat.fill(0.0)
    var.s_shpkdzl_hat.fill(0.0)
    var.Jacobil.fill(0.0)
    var.JacobiInvl.fill(0.0)
    for ik in range(var.s_mgk[None]):
        xL = ti.Vector([0.0, 0.0, 0.0, 0.0])
        yL = ti.Vector([0.0, 0.0, 0.0, 0.0])
        zL = ti.Vector([0.0, 0.0, 0.0, 0.0])
        jLoop = params.s_maxNode[None]
        ie = var.s_gp_ele[0, ik] - 1
        iLocint = var.s_gp_ele[1, ik]
        xsi = var.gp_loc2D[0, iLocint - 1]
        eta = var.gp_loc2D[1, iLocint - 1]
        shapef = evl_FEM_shape4(xsi, eta)
        dshape_xsi, dshape_eta = evl_FEM_dshape4(xsi, eta)
        for jp in range(jLoop):
            var.s_shpk[jp, ik] = shapef[jp]
            inode = var.s_nele_node[jp + 1, ie] - 1
            Xg = var.s_xm[inode] - var.s_xmk[ik]

            xL[jp] = (var.s_q_gl0[ik] @ Xg)[0]
            yL[jp] = (var.s_q_gl0[ik] @ Xg)[1]
            zL[jp] = (var.s_q_gl0[ik] @ Xg)[2]

        var.Jacobil[ik][0, 0] = dshape_xsi.dot(xL)
        var.Jacobil[ik][0, 1] = dshape_xsi.dot(yL)
        var.Jacobil[ik][1, 0] = dshape_eta.dot(xL)
        var.Jacobil[ik][1, 1] = dshape_eta.dot(yL)
        var.JacobiInvl[ik] = var.Jacobil[ik].inverse()

        for zn in range(s_maxntg):
            for jp in range(jLoop):
                var.s_shpkdxl[jp, ik, zn] = var.JacobiInvl[ik][0, 0] * dshape_xsi[jp] + var.JacobiInvl[ik][0, 1] * dshape_eta[jp]
                var.s_shpkdyl[jp, ik, zn] = var.JacobiInvl[ik][1, 0] * dshape_xsi[jp] + var.JacobiInvl[ik][1, 1] * dshape_eta[jp]
                var.s_shpkdxl_hat[jp, ik, zn] = gp_loc1D[zn] * var.s_shpkdxl[jp, ik, zn]
                var.s_shpkdyl_hat[jp, ik, zn] = gp_loc1D[zn] * var.s_shpkdyl[jp, ik, zn]
                var.s_shpkdzl_hat[jp, ik, zn] = shapef[jp] * 2 / var.s_thkk[ik]

    print('*** cal shape function for all node passed.')