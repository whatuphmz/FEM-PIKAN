import numpy as np
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Sh_material_initial import mat
import taichi as ti

@ti.kernel
def get_initial_force():
    # Initial displacement & velocity fields of nodes
    var.s_complete_damage.fill(0)
    var.s_ibtype.fill(0)
    var.s_damage.fill(0)
    var.Fsext.fill(0.0)
    var.Msext.fill(0.0)
    var.Fsint.fill(0.0)
    var.Msint.fill(0.0)
    var.s_disp.fill(0.0)
    var.s_vel.fill(0.0)
    var.s_acc.fill(0.0)
    var.s_angdisp.fill(0.0)
    var.s_angvel.fill(0.0)
    var.s_angacc.fill(0.0)
    var.s_disp_last.fill(0.0)
    var.s_vel_last.fill(0.0)
    var.s_acc_last.fill(0.0)
    var.s_angdisp_last.fill(0.0)
    var.s_angvel_last.fill(0.0)
    var.s_angacc_last.fill(0.0)
    # # Initial velocity
    # for ip in range(var.s_nnode[None]):
        # var.s_vel[ip][1] = -3.5 * ti.sin(ti.radians(20)) * 1
        # var.s_vel[ip][2] = 3.5 * ti.cos(ti.radians(20)) * 1
        # var.s_vel_last[ip][1] = -3.5 * ti.sin(ti.radians(20)) * 1
        # var.s_vel_last[ip][2] = 3.5 * ti.cos(ti.radians(20)) * 1
    var.s_effEopt.fill(0.0)

    # Initialize kinematic parameters of Gauss points
    var.s_pressure.fill(0.0)
    var.s_damagek.fill(0)
    
    var.s_dispk.fill(0.0)
    var.s_velk.fill(0.0)
    var.s_angvelk.fill(0.0)
    var.s_angdispk.fill(0.0)
    
    # for ik in range(var.s_mgk[None]):    
        # var.s_velk[ik][1] = -3.5 * ti.sin(ti.radians(20)) * 1
        # var.s_velk[ik][2] = 3.5 * ti.cos(ti.radians(20)) * 1

    # Initial stress states
    var.s_stsgp.fill(0.0)
    var.s_back_stress_gp.fill(0.0)
    var.s_plastic_strain_gp.fill(0.0)
    var.s_effstsgp.fill(0.0)
    var.s_effstegp.fill(0.0)
    for ik in range(var.s_mgk[None]):    
            # Fill effEopt for weighted average
            jLoop = params.s_maxNode[None]
            for jp in range(jLoop):
                jpt = var.s_nele_node[jp + 1, ik]
                var.s_effEopt[jpt] += var.s_shpk[jp, ik]
@ti.kernel
def get_initial_condition():
    # Initial Acceleration field (Fint = 0 at time = 0)
    for ip in range(var.s_nnode[None]):
        var.s_acc[ip] = var.s_mass_1[ip] * var.Fsext[ip]
        var.s_angacc[ip] = var.s_inertia_1[ip] * var.Msext[ip]
        var.s_acc_last[ip] = var.s_acc[ip]
        var.s_angacc_last[ip] = var.s_angacc[ip]

    # Initialize elastoplastic parameters
    for ik in range(var.s_mgk[None]):
        for ig in range(params.s_maxntg[None]):
            var.s_yield_sts_gp[ig, ik] = mat.pl_k0[mat.s_MatTypek[ik]]  # Initial yield stress
            var.s_damagek_sec[ig, ik] = 0.0

    if params.s_crack_grow[None] or params.s_ini_crack[None]:
        for ik in range(var.s_mgk[None]):
            if var.s_complete_damagek[ik] != 1:
                var.s_complete_damagek[ik] = 0
        var.s_crack_tip_num[None] = 0
        for ik in range(var.s_mgk[None]):
            if var.s_complete_damagek[ik] == 1:
                for ig in range(params.s_maxntg[None]):
                    var.s_damagek_sec[ig, ik] = mat.lm_D_c[mat.s_MatTypek[ik]]
                var.s_crack_tip_num[None] += 1
                var.s_crack_tip_list[var.s_crack_tip_num[None]] = ik

        var.s_crack_tip_block_num[None] = 1