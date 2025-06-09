import numpy as np
import taichi as ti
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Gaussian import *
from FEM.Evl_fem_shape4 import *

@ti.kernel
def Sh_coordinate_system_stiff():
    # Loop over Gauss points
    for ik in range(var.s_mgk[None]):
        ie = var.s_gp_ele[0, ik] - 1
        iLocint = var.s_gp_ele[1, ik]
        
        # Get Gauss point coordinates
        xsi = var.gp_loc2D[0, iLocint-1]
        eta = var.gp_loc2D[1, iLocint-1]
        if params.s_maxNode[None] == 4:
            
            matrix = ti.Matrix.rows([var.s_xm[var.s_nele_node[1, ie] - 1],
                     var.s_xm[var.s_nele_node[2, ie] - 1],
                     var.s_xm[var.s_nele_node[3, ie] - 1],
                     var.s_xm[var.s_nele_node[4, ie] - 1]])
            # Evaluate FEM shape functions and derivatives at the Gauss point
            shapef = evl_FEM_shape4(xsi, eta)
            dshape_xsi, dshape_eta = evl_FEM_dshape4(xsi, eta)
            
            # Calculate the coordinates of the Gauss point in the element's local coordinate system
            var.s_xmk[ik] = shapef @ matrix
            
            # Calculate derivatives of shape functions with respect to xsi and eta
            e_1x = (dshape_xsi @ matrix)[0]
            e_1y = (dshape_xsi @ matrix)[1]
            e_1z = (dshape_xsi @ matrix)[2]
            
            e_2x = (dshape_eta @ matrix)[0]
            e_2y = (dshape_eta @ matrix)[1]
            e_2z = (dshape_eta @ matrix)[2]

            # Normalize vectors e_1, e_2, and e_3
            e1_Eulicd = ti.sqrt(e_1x**2 + e_1y**2 + e_1z**2)
            var.s_e_la1[ik] = ti.Vector([e_1x / e1_Eulicd, e_1y / e1_Eulicd, e_1z / e1_Eulicd])
            
            e_1x = var.s_e_la1[ik][0]
            e_1y = var.s_e_la1[ik][1]
            e_1z = var.s_e_la1[ik][2]
            
            # Calculate e_3 as the cross product of e_1 and e_2
            e_3x = e_1y * e_2z - e_1z * e_2y
            e_3y = e_1z * e_2x - e_1x * e_2z
            e_3z = e_1x * e_2y - e_1y * e_2x
            
            e3_Eulicd = ti.sqrt(e_3x**2 + e_3y**2 + e_3z**2)
            var.s_e_la3[ik] = ti.Vector([e_3x / e3_Eulicd, e_3y / e3_Eulicd, e_3z / e3_Eulicd])
            
            # Calculate e_2 as the cross product of e_3 and e_1
            var.s_e_la2[ik] = ti.Vector([var.s_e_la3[ik][1] * var.s_e_la1[ik][2] - var.s_e_la3[ik][2] * var.s_e_la1[ik][1],
                                         var.s_e_la3[ik][2] * var.s_e_la1[ik][0] - var.s_e_la3[ik][0] * var.s_e_la1[ik][2],
                                         var.s_e_la3[ik][0] * var.s_e_la1[ik][1] - var.s_e_la3[ik][1] * var.s_e_la1[ik][0]])
            
            # Store the transformation matrix (global -> lamina)
            var.s_q_gl[ik][0, :] = var.s_e_la1[ik]
            var.s_q_gl[ik][1, :] = var.s_e_la2[ik]
            var.s_q_gl[ik][2, :] = var.s_e_la3[ik]
     
    #!c==========================================c
    #!c      Search node number around node.     c
    #!c==========================================c
    #!c...不考虑加筋的时候,节点初始的fiber方向是通过节点cell的边叉乘求得的,所以需要按逆时针排列好.
    #!c...现在节点初始fiber方向通过周围单元法向平均求得,所以不需要对节点周围节点进行拓扑排序.
    #!c
    if var.istep[None] == 0:
        # Part 1: Search node number around node
        for ip in range(var.s_nnode[None]):
            var.s_node_ele[0, ip] = ip + 1
            var.s_node_node[0, ip] = ip + 1
            
            k = 0
            for ie in range(var.s_nelement[None]):
                for jp in ti.static(range(4)):
                    if var.s_nele_node[jp + 1, ie] == ip + 1:
                        var.s_node_ele[k + 2, ip] = ie + 1
                        k += 1
            var.s_node_ele[1, ip] = k  # Number of elements sharing the node
    
        # !c
        # !c...(2)...Calculate the initial fiber direction, There are two method: see the stiff plate document
        # !c
        fib = ti.Vector([0.0, 0.0, 0.0])
        for ip in range(var.s_nnode[None]):
            fib.fill(0.0)
            for i in range(var.s_node_ele[1, ip]):
                ie = var.s_node_ele[i + 2, ip]
                if params.s_maxInt[None] == 1:
                    ik = var.s_ele_gp[1, ie-1]
                    fib += var.s_e_la3[ik-1]
                else:
                    for knum in range(var.s_ele_gp[0, ie-1]):
                        ik = var.s_ele_gp[knum + 1, ie-1]
                        fib += var.s_e_la3[ik-1]
            
            # Normalize fiber vector (Euclidean norm)
            e3_Eulicd = ti.sqrt(fib[0]**2 + fib[1]**2 + fib[2]**2)
            var.s_e_fi3[ip] = ti.Vector([fib[0] / e3_Eulicd, fib[1] / e3_Eulicd, fib[2] / e3_Eulicd])
    
        print('*** Shell fiber direction has been computed!')
    
        #!c
        #!c...(3)...Calculate the fiber coordinate system
        #!c
        for ip in range(var.s_nnode[None]):
            e_3x, e_3y, e_3z = var.s_e_fi3[ip][0], var.s_e_fi3[ip][1], var.s_e_fi3[ip][2]
            e_2x, e_2y, e_2z = 0.0, 0.0, 0.0
            
            if e_3z > e_3x and e_3z > e_3y:
                e_2x, e_2y, e_2z = 0.0, e_3z, -e_3y
            elif e_3x > e_3y and e_3x > e_3z:
                e_2x, e_2y, e_2z = -e_3z, 0.0, e_3x
            else:
                e_2x, e_2y, e_2z = e_3y, -e_3x, 0.0
            
            e2_Eulicd = ti.sqrt(e_2x**2 + e_2y**2 + e_2z**2)
            var.s_e_fi2[ip] = ti.Vector([e_2x / e2_Eulicd, e_2y / e2_Eulicd, e_2z / e2_Eulicd])
            var.s_e_fi1[ip] = ti.Vector([e_2y / e2_Eulicd * e_3z - e_2z / e2_Eulicd * e_3y, e_2z / e2_Eulicd * e_3x - e_2x / e2_Eulicd * e_3z, e_2x / e2_Eulicd * e_3y - e_2y / e2_Eulicd * e_3x])
            var.s_e_fi1_t0[ip] = var.s_e_fi1[ip]
            var.s_e_fi2_t0[ip] = var.s_e_fi2[ip]
            var.s_e_fi3_t0[ip] = var.s_e_fi3[ip]
            # !c
            # !c....q_fl*r_f=r_la : fiber ==> lamina
            # !c
            # !c###############[wait for complete]#################
            # !c
            # !c....s_q_fg*r_f=r_lg : fiber ==> global
            # !c
            var.s_q_fg[ip][0, :] = var.s_e_fi1[ip]
            var.s_q_fg[ip][1, :] = var.s_e_fi2[ip]
            var.s_q_fg[ip][2, :] = var.s_e_fi3[ip]

        print("*** Fiber coordinate system has been computed!")