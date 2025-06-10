import numpy as np
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Sh_material_initial import mat
import taichi as ti

# !c===============================================c
# !c     Calculate the external force vector       c
# !c===============================================c
# !c
@ti.kernel
def Sh_external_force():

    # 初始化外力
    var.Fsext.fill(0.0)
    var.Msext.fill(0.0)
    var.Fs_body.fill(0.0)
    var.Ms_body.fill(0.0)
    var.Fs_press.fill(0.0)
    var.Ms_press.fill(0.0)
    var.Fs_edge.fill(0.0)
    var.Ms_edge.fill(0.0)
    var.Fs_hourg.fill(0.0)
    var.Ms_hourg.fill(0.0)

    # 设置重力加速度
    var.s_uf_body.fill([0.0, 0.0, -10000.0])
    
    for ik in range(var.s_mgk[None]):
        mLoop = params.s_maxNode[None]
        for ip in range(mLoop):
            ipt = var.s_nele_node[ip + 1, ik] - 1
            var.Fs_body[ipt] += var.s_dvomk[ik] * mat.el_rho[mat.s_MatTypek[ik]] * var.s_shpk[ip, ik] * var.s_uf_body[ik]
            # 计算体积力矩，假设为 0
            var.Ms_body[ipt] += [0.0, 0.0, 0.0]

    # 计算表面压力形成的力和力矩
    var.s_pressure.fill(0.0)
    for ik in range(var.s_mgk[None]):
        if var.s_xmk0[ik][2] > -0.01:
            mLoop = params.s_maxNode[None]
            s_hpress = var.s_pressure[ik] * var.s_e_la3[ik]

            for ip in range(mLoop):
                ipt = var.s_nele_node[ip + 1, ik] - 1
                var.Fs_press[ipt] += var.s_dareak[ik] * s_hpress * var.s_shpk[ip, ik]
                # 表面压力的力矩近似为零
                var.Ms_press[ipt] += [0.0, 0.0, 0.0]

    # # 计算边缘力和力矩
    var.s_hedge_force.fill([0.0, 0.0, 0.0])
    var.s_hedge_moment.fill([0.0, 0.0, 0.0])
    for ip in range(var.s_nnode[None]):
        if var.s_xm0[ip][1] > 0.495:
            var.s_hedge_force[ip] = [0.0, 0.0, 0.0]

    for ip in range(var.s_nnode[None]):
        # 计算小时力和力矩
        var.Fsext[ip] = var.Fs_body[ip] + var.Fs_press[ip] + var.Fs_edge[ip] - var.Fs_hourg[ip] + var.s_hedge_force[ip]
        var.Msext[ip] = var.Ms_body[ip] + var.Ms_press[ip] + var.Ms_edge[ip] - var.Ms_hourg[ip] + var.s_hedge_moment[ip]
