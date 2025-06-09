import numpy as np
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.ReadALine import *
import taichi as ti
  
@ti.kernel
def Sh_essential_bound():
    angular_same = 1
    var.s_nDispbcX[None] = 0
    var.s_nDispbcY[None] = 0
    var.s_nDispbcZ[None] = 0
    var.s_nVelbcX[None]  = 0
    var.s_nVelbcY[None]  = 0
    var.s_nVelbcZ[None]  = 0
    var.s_nAngDispbcX[None] = 0
    var.s_nAngDispbcY[None] = 0
    var.s_nAngDispbcZ[None] = 0
    var.s_nAngVelbcX[None]  = 0
    var.s_nAngVelbcY[None]  = 0
    var.s_nAngVelbcZ[None]  = 0
    var.s_nfree[None] = 0
    # Loop over nodes and set boundary conditions
    for ip in range(var.s_nnode[None]):
        if (abs(abs(var.s_xm0[ip][0]) - 0.5) < 0.01) or (abs(abs(var.s_xm0[ip][1]) - 0.5) < 0.01):
            var.s_ibtype[ip] = 200
            
            # Displacement boundary (simple support)
            var.s_nDispbcX[None] += 1
            var.s_LmDispbcX[var.s_nDispbcX[None]] = ip
            var.s_vDispbcX[var.s_nDispbcX[None]] = 0.0
            var.s_nDispbcY[None] += 1
            var.s_LmDispbcY[var.s_nDispbcY[None]] = ip
            var.s_vDispbcY[var.s_nDispbcY[None]] = 0.0
            var.s_nDispbcZ[None] += 1
            var.s_LmDispbcZ[var.s_nDispbcZ[None]] = ip
            var.s_vDispbcZ[var.s_nDispbcZ[None]] = 0.0
            
            if angular_same == 1:
                var.s_nAngDispbcX[None] = var.s_nDispbcX[None]
                var.s_LmAngDispbcX[var.s_nAngDispbcX[None]] = var.s_LmDispbcX[var.s_nDispbcX[None]]
                var.s_vAngDispbcX[var.s_nAngDispbcX[None]] = 0.0
                var.s_nAngDispbcY[None] = var.s_nDispbcY[None]
                var.s_LmAngDispbcY[var.s_nAngDispbcY[None]] = var.s_LmDispbcY[var.s_nDispbcY[None]]
                var.s_vAngDispbcY[var.s_nAngDispbcY[None]] = 0.0
                var.s_nAngDispbcZ[None] = var.s_nDispbcZ[None]
                var.s_LmAngDispbcZ[var.s_nAngDispbcZ[None]] = var.s_LmDispbcZ[var.s_nDispbcZ[None]]
                var.s_vAngDispbcZ[var.s_nAngDispbcZ[None]] = 0.0
        else:
            var.s_nfree[None] += 1
    
    print('*** Essential boundary condition node & value set got.')