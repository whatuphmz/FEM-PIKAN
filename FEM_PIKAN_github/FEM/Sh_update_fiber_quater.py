import numpy as np
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Sh_material_initial import mat
from FEM.Gaussian import *
import taichi as ti

# !c==========================================c
# !c   calculate the fiber direction          c
# !c                at each step              c
# !c==========================================c
# !c
@ti.kernel
def Sh_update_fiber_quater():
    s_e_fi_last3 = ti.Vector([0.0, 0.0, 0.0])
    eps_temp = 1.0e-12
    quat_fi3 = ti.Vector([0.0, 0.0, 0.0, 0.0])
    quat_rota = ti.Vector([0.0, 0.0, 0.0, 0.0])
    quat_rota_conj = ti.Vector([0.0, 0.0, 0.0, 0.0])
    quat_fi3_temp = ti.Vector([0.0, 0.0, 0.0, 0.0])


    for ipt in range(var.s_nnode[None]):
        if var.s_complete_damage[ipt] != 1:
            # 计算 dtheta_norm（变形量的大小）
            dtheta_norm = ti.sqrt(var.s_dangdisp[ipt][0]**2 + var.s_dangdisp[ipt][1]**2 + var.s_dangdisp[ipt][2]**2)

            if dtheta_norm < eps_temp:
                var.s_delta_n[ipt] = [0.0, 0.0, 0.0]
                var.s_ddelta_n[ipt] = [0.0, 0.0, 0.0]
            else:
                # 构建四元数 quat_fi3（上一步的fiber方向）
                quat_fi3 = [0.0, var.s_e_fi3[ipt][0], var.s_e_fi3[ipt][1], var.s_e_fi3[ipt][2]]

                # 构建旋转四元数 quat_rota
                quat_rota[0] = ti.cos(0.5 * dtheta_norm)
                quat_rota[1] = ti.sin(0.5 * dtheta_norm) * var.s_dangdisp[ipt][0] / dtheta_norm
                quat_rota[2] = ti.sin(0.5 * dtheta_norm) * var.s_dangdisp[ipt][1] / dtheta_norm
                quat_rota[3] = ti.sin(0.5 * dtheta_norm) * var.s_dangdisp[ipt][2] / dtheta_norm

                # 计算旋转四元数的共轭
                quat_rota_conj[0] = quat_rota[0]
                quat_rota_conj[1] = -quat_rota[1]
                quat_rota_conj[2] = -quat_rota[2]
                quat_rota_conj[3] = -quat_rota[3]

                # 四元数相乘，计算旋转后的fiber方向
                quat_fi3_temp = mult_quater(quat_rota, quat_fi3)
                quat_fi3 = mult_quater(quat_fi3_temp, quat_rota_conj)

                # 更新 fiber 方向
                s_e_fi_last3 = var.s_e_fi3[ipt]
                var.s_e_fi3[ipt][0] = quat_fi3[1]
                var.s_e_fi3[ipt][1] = quat_fi3[2]
                var.s_e_fi3[ipt][2] = quat_fi3[3]

                # 计算 s_delta_n 和 s_ddelta_n
                var.s_delta_n[ipt][0] = var.s_e_fi3[ipt][0] - var.s_e_fi3_t0[ipt][0]
                var.s_delta_n[ipt][1] = var.s_e_fi3[ipt][1] - var.s_e_fi3_t0[ipt][1]
                var.s_delta_n[ipt][2] = var.s_e_fi3[ipt][2] - var.s_e_fi3_t0[ipt][2]

                var.s_ddelta_n[ipt][0] = var.s_e_fi3[ipt][0] - s_e_fi_last3[0]
                var.s_ddelta_n[ipt][1] = var.s_e_fi3[ipt][1] - s_e_fi_last3[1]
                var.s_ddelta_n[ipt][2] = var.s_e_fi3[ipt][2] - s_e_fi_last3[2]              
        else:
            # 如果损伤已完成，直接更新
            var.s_delta_n[ipt][0] = var.s_e_fi3[ipt][0] - var.s_e_fi3_t0[ipt][0]
            var.s_delta_n[ipt][1] = var.s_e_fi3[ipt][1] - var.s_e_fi3_t0[ipt][1]
            var.s_delta_n[ipt][2] = var.s_e_fi3[ipt][2] - var.s_e_fi3_t0[ipt][2]

            var.s_ddelta_n[ipt][0] = 0.0
            var.s_ddelta_n[ipt][1] = 0.0
            var.s_ddelta_n[ipt][2] = 0.0

@ti.func
def mult_quater(q_in_1, q_in_2):
    q_out = ti.Vector([0.0, 0.0, 0.0, 0.0])

    q_out[0] = q_in_1[0] * q_in_2[0] - q_in_1[1] * q_in_2[1] - q_in_1[2] * q_in_2[2] - q_in_1[3] * q_in_2[3]
    q_out[1] = q_in_1[1] * q_in_2[0] + q_in_1[0] * q_in_2[1] + q_in_1[2] * q_in_2[3] - q_in_1[3] * q_in_2[2]
    q_out[2] = q_in_1[2] * q_in_2[0] + q_in_1[0] * q_in_2[2] + q_in_1[3] * q_in_2[1] - q_in_1[1] * q_in_2[3]
    q_out[3] = q_in_1[3] * q_in_2[0] + q_in_1[0] * q_in_2[3] + q_in_1[1] * q_in_2[2] - q_in_1[2] * q_in_2[1]

    return q_out