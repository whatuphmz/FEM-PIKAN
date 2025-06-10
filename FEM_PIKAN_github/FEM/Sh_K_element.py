from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Sh_material_initial import mat
from FEM.Gaussian import *
from FEM.Evl_fem_shape4 import *
import taichi as ti

# !c================================================c
# !c   Compute element stiffness matrices for       c
# !c        shell FEM using Gaussian quadrature     c
# !c================================================c
# !c
@ti.kernel
def Sh_K_element():
    s_maxntg = params.s_maxntg[None]
    s_maxNode = params.s_maxNode[None]
    var.s_Kvv.fill(0.0)
    var.s_Kvw.fill(0.0)
    var.s_Kwv.fill(0.0)
    var.s_Kww.fill(0.0)
    var.Bv.fill(0.0)
    var.Bw.fill(0.0)
    var.Bv_T.fill(0.0)
    var.Bw_T.fill(0.0)
    ntg = s_maxntg
    gp_loc1D, gp_weight1D = gauss1D(ntg)

    for ik in range(var.s_mgk[None]):
        nTmp = ti.Vector([0.0, 0.0, 1.0])
        iMat = mat.s_MatTypek[ik]
        tmp = 0.0
        C = ti.Matrix.zero(float, 6, 6)
        tmp = mat.el_E[iMat] / (1.0 - mat.el_V[iMat]**2)
        C[0,0] = tmp
        C[0,1] = tmp * mat.el_V[iMat]
        C[1,0] = C[0,1]
        C[1,1] = tmp
        C[3,3] = tmp * (1.0 - mat.el_V[iMat]) / 2.0
        C[4,4] = C[3,3] * 5.0 / 6.0
        C[5,5] = C[4,4]

        for zn in range(ntg):
            z_temp = 0.5 * var.s_thkk[ik] * gp_loc1D[zn]

            for jp in range(s_maxNode):
                ip = var.s_nele_node[jp + 1, ik] - 1
        
                var.Bv[jp, ik, zn][0, 0] = var.s_shpkdxl[jp, ik, zn]
                var.Bv[jp, ik, zn][1, 1] = var.s_shpkdyl[jp, ik, zn]
                var.Bv[jp, ik, zn][2, 2] = var.s_shpkdzl[jp, ik, zn]
                var.Bv[jp, ik, zn][3, 0] = var.s_shpkdyl[jp, ik, zn]
                var.Bv[jp, ik, zn][3, 1] = var.s_shpkdxl[jp, ik, zn]
                var.Bv[jp, ik, zn][4, 1] = var.s_shpkdzl[jp, ik, zn]
                var.Bv[jp, ik, zn][4, 2] = var.s_shpkdyl[jp, ik, zn]
                var.Bv[jp, ik, zn][5, 0] = var.s_shpkdzl[jp, ik, zn]
                var.Bv[jp, ik, zn][5, 2] = var.s_shpkdxl[jp, ik, zn]
        
                # var.Bw[jp][0, 0] = 0.0
                var.Bw[jp, ik, zn][1, 0] = -var.s_shpkdyl_hat[jp, ik, zn] * nTmp[2]
                var.Bw[jp, ik, zn][2, 0] = var.s_shpkdzl_hat[jp, ik, zn] * nTmp[1]
                var.Bw[jp, ik, zn][3, 0] = -var.s_shpkdxl_hat[jp, ik, zn] * nTmp[2]
                var.Bw[jp, ik, zn][4, 0] = (var.s_shpkdyl_hat[jp, ik, zn] * nTmp[1] - var.s_shpkdzl_hat[jp, ik, zn] * nTmp[2])
                var.Bw[jp, ik, zn][5, 0] = var.s_shpkdxl_hat[jp, ik, zn] * nTmp[1]

                var.Bw[jp, ik, zn][0, 1] = var.s_shpkdxl_hat[jp, ik, zn] * nTmp[2]
                # var.Bw[jp][1, 1] = 0.0
                var.Bw[jp, ik, zn][2, 1] = -var.s_shpkdzl_hat[jp, ik, zn] * nTmp[0]
                var.Bw[jp, ik, zn][3, 1] = var.s_shpkdyl_hat[jp, ik, zn] * nTmp[2]
                var.Bw[jp, ik, zn][4, 1] = -var.s_shpkdyl_hat[jp, ik, zn] * nTmp[0]
                var.Bw[jp, ik, zn][5, 1] = (var.s_shpkdzl_hat[jp, ik, zn] * nTmp[2] - var.s_shpkdxl_hat[jp, ik, zn] * nTmp[0])

                var.Bw[jp, ik, zn][0, 2] = -var.s_shpkdxl_hat[jp, ik, zn] * nTmp[1]
                var.Bw[jp, ik, zn][1, 2] = var.s_shpkdyl_hat[jp, ik, zn] * nTmp[0]
                # var.Bw[jp][2, 2] = 0.0
                var.Bw[jp, ik, zn][3, 2] = (var.s_shpkdxl_hat[jp, ik, zn] * nTmp[0] - var.s_shpkdyl_hat[jp, ik, zn] * nTmp[1])
                var.Bw[jp, ik, zn][4, 2] = var.s_shpkdzl_hat[jp, ik, zn] * nTmp[0]
                var.Bw[jp, ik, zn][5, 2] = -var.s_shpkdzl_hat[jp, ik, zn] * nTmp[1]

                var.Bw[jp, ik, zn] *= 0.5 * var.s_thkk[ik]
                var.Bv_T[jp, ik, zn] = var.Bv[jp, ik, zn].transpose()
                var.Bw_T[jp, ik, zn] = var.Bw[jp, ik, zn].transpose()

            for ii in range(s_maxNode):
                for jj in range(s_maxNode):
                    var.s_Kvv[ii, jj, ik] += var.s_q_gl0[ik].transpose() @ (0.5 * var.s_thkk[ik] * var.s_dareak[ik] * ((var.Bv_T[ii, ik, zn] @ C) @ var.Bv[jj, ik, zn]) * gp_weight1D[zn]) @ var.s_q_gl0[ik]
                    var.s_Kvw[ii, jj, ik] += var.s_q_gl0[ik].transpose() @ (0.5 * var.s_thkk[ik] * var.s_dareak[ik] * ((var.Bv_T[ii, ik, zn] @ C) @ var.Bw[jj, ik, zn]) * gp_weight1D[zn]) @ var.s_q_gl0[ik]
                    var.s_Kwv[ii, jj, ik] += var.s_q_gl0[ik].transpose() @ (0.5 * var.s_thkk[ik] * var.s_dareak[ik] * ((var.Bw_T[ii, ik, zn] @ C) @ var.Bv[jj, ik, zn]) * gp_weight1D[zn]) @ var.s_q_gl0[ik]
                    var.s_Kww[ii, jj, ik] += var.s_q_gl0[ik].transpose() @ (0.5 * var.s_thkk[ik] * var.s_dareak[ik] * ((var.Bw_T[ii, ik, zn] @ C) @ var.Bw[jj, ik, zn]) * gp_weight1D[zn]) @ var.s_q_gl0[ik]

