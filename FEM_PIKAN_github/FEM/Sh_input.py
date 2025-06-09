import taichi as ti
from FEM.Sh_parameter import params
from FEM.Sh_cache import var
from FEM.Sh_material_initial import mat
# from FEM.Sh_coordinate_system import Sh_coordinate_system
from FEM.Sh_coordinate_system_stiff import *
from FEM.Sh_direct_find import *
from FEM.Sh_initial_condition import *
from FEM.Sh_external_force import *
from FEM.Sh_Essential_bound import *
from FEM.Sh_output_paraview_ascii import Sh_output_paraview_ascii
from FEM.ReadALine import *
from FEM.Gaussian import *
from FEM.Evl_fem_shape4 import *
from FEM.ajacob import *


@ti.data_oriented
class Sh_input:
    def __init__(self):
        # !c==========================================c
        # !c        Get the initial information       c
        # !c==========================================c
        # !c
        fhead = ''
        frname = ''
        i_ask_fhead = 0
        var.s_nintElem[None] = params.s_maxIntElem[None]
        var.s_ele_node[None] = params.s_maxNode[None]
        #!c==========================================c
        #!c  set the file that used in this program  c
        #!c==========================================c
        #!c
        #!c...fhead is input from the command line
        #!c
        s_maxNode = params.s_maxNode[None]
        s_Istage = params.s_Istage[None]
        
        fhead='InputModel/NumericalParameterSetting'
        frname = appext(fhead, 'inp')
        fwname_echo = appext(fhead, 'echo')
        params.s_ifwEcho = open(fwname_echo, 'w')
        params.si_ifrInp = open(frname, 'r')
        # 读取节点和单元数量
        line = read_a_line(params.si_ifrInp, params.s_ifwEcho)
        var.s_nnode[None], var.s_nelement[None] = map(int, line.split())
        print(f"*** The number of structure nodes is: {var.s_nnode[None]}")
        print(f"*** The number of structure elements is: {var.s_nelement[None]}")
        params.s_ifwEcho.write(f"{var.s_nnode[None]} {var.s_nelement[None]}\n")
        params.s_ifwEcho.close()
        params.si_ifrInp.close()

        # File prompt handling
        if i_ask_fhead == 1:
            fhead = input("Please input the head of the files name: ")
        else:
            # Set the default file head (For example: 'box_stone_1d5mm')
            fhead = 'InputModel/plate_0.05'

        # Assign file names with appended extensions
        frname = appext(fhead, 'inp')
        print(f"frname= {frname}")
        fwname_echo = appext(fhead, 'echo')

        params.s_ifrInp = open(frname, 'r')
        params.s_ifwEcho = open(fwname_echo, 'w')

        for ip in range(var.s_nnode[None]):
            line = read_a_line(params.s_ifrInp, params.s_ifwEcho)
            num, xxx, yyy, zzz = map(float, line.split())
            xtemp = xxx
            ytemp = yyy
            ztemp = zzz
            var.s_xm[ip] = ti.Vector([xtemp, ytemp, ztemp])
            params.s_ifwEcho.write(f"{ip + 1:8d}{var.s_xm[ip][0]:15.6e}{var.s_xm[ip][1]:15.6e}{var.s_xm[ip][2]:15.6e}\n")

        for ie in range(var.s_nelement[None]):
            line = read_a_line(params.s_ifrInp, params.s_ifwEcho)
            for i, val in enumerate(map(int, line.split())):
                var.s_nele_node[i, ie] = val
            line_values = [str(var.s_nele_node[i, ie]) for i in ti.static(range(5))]
            params.s_ifwEcho.write(" ".join(line_values) + "\n")

        self.get_input1()

        s_stiffness = params.s_stiffness[None]
        if s_stiffness == 1:
            Sh_coordinate_system_stiff()
        # else:
        #     Sh_coordinate_system()

        self.get_input2()

        Sh_direct_find()

        self.get_input3()
        
        get_initial_force()
        Sh_external_force()
        get_initial_condition()

        Sh_essential_bound()
        if s_Istage == 1:
            Sh_output_paraview_ascii()
        params.s_ifrInp.close()
        params.s_ifwEcho.close()
        

    # !c==========================================c
    # !c           Gauss quadrature pt.           c
    # !c                                          c
    # !c  nint: gauss pt. number along            c
    # !c        1 direction of each element       c
    # !c  nintElem: gauss pt. number              c
    # !c              of one element              c
    # !c  ele_node: node number of per element    c
    # !c==========================================c
    # !c
    @ti.kernel
    def get_input1(self):
        gauss2D(var.s_ele_node, var.s_nintElem)  # 获取2D高斯点

        xyloc_of_elem4 = ti.Matrix([[-1.0, 1.0, 1.0, -1.0],
                                   [-1.0, -1.0, 1.0, 1.0]])

        for ie in range(var.s_nelement[None]):
            # 获取节点坐标
            xn1 = var.s_xm[var.s_nele_node[1, ie] - 1][0]
            yn1 = var.s_xm[var.s_nele_node[1, ie] - 1][1]
            zn1 = var.s_xm[var.s_nele_node[1, ie] - 1][2]
            xn2 = var.s_xm[var.s_nele_node[2, ie] - 1][0]
            yn2 = var.s_xm[var.s_nele_node[2, ie] - 1][1]
            zn2 = var.s_xm[var.s_nele_node[2, ie] - 1][2]
            xn3 = var.s_xm[var.s_nele_node[3, ie] - 1][0]
            yn3 = var.s_xm[var.s_nele_node[3, ie] - 1][1]
            zn3 = var.s_xm[var.s_nele_node[3, ie] - 1][2]
            xn4, yn4, zn4 = 0.0, 0.0, 0.0
            if var.s_ele_node[None] == 4:
                xn4 = var.s_xm[var.s_nele_node[4, ie] - 1][0]
                yn4 = var.s_xm[var.s_nele_node[4, ie] - 1][1]
                zn4 = var.s_xm[var.s_nele_node[4, ie] - 1][2]

            matrix = ti.Matrix.rows([var.s_xm[var.s_nele_node[1, ie] - 1],
                                     var.s_xm[var.s_nele_node[2, ie] - 1],
                                     var.s_xm[var.s_nele_node[3, ie] - 1],
                                     var.s_xm[var.s_nele_node[4, ie] - 1]])

            for iLocint in range(var.s_nintElem[None]):
                var.s_gp_ele[0, ie] = ie + 1
                var.s_gp_ele[1, ie] = iLocint + 1

                var.s_ele_gp[0, ie] = var.s_nintElem[None]
                var.s_ele_gp[iLocint+1, ie] = ie + 1

                xsi = var.gp_loc2D[0, iLocint]
                eta = var.gp_loc2D[1, iLocint]

                shapef = evl_FEM_shape4(xsi, eta)
                var.s_xmk[ie] = shapef @ matrix
                
                # 计算雅可比矩阵
                ajj = ajacob4_space(xn1, xn2, xn3, xn4, yn1, yn2, yn3, yn4, zn1, zn2, zn3, zn4, xsi, eta)
                var.s_dareak[ie] = ajj * var.gp_weight2D[iLocint]  # 高斯点权重

                if var.s_ele_node[None] == 4:
                    for inode in range(1, var.s_ele_node[None] + 1):
                        jnode = var.s_nele_node[inode,ie]
            
                        # Call the Jacobian matrix function (assuming it's defined elsewhere)
                        ajj = ajacob4_space(xn1, xn2, xn3, xn4, yn1, yn2, yn3, yn4, 
                                            zn1, zn2, zn3, zn4, 
                                            xyloc_of_elem4[0,inode - 1],
                                            xyloc_of_elem4[1,inode - 1])
            
                        # Update the area and volume for the node
                        var.s_darea[jnode - 1] += ajj

        var.s_mgk[None] = var.s_nelement[None]  # 高斯点的总数
        print(f"*** New s_mgk: {var.s_mgk[None]:7d}")  # 输出高斯点数量
        var.s_ntotal[None] = var.s_mgk[None] + var.s_nnode[None]  # 总的节点数

        print('c=========================================================c')
        print(f'c       Initial particle configuration generated         c')
        print(f'c         Total number of particles {var.s_ntotal[None]} c')
        print('c=========================================================c')

    # ============================================
    # Calculate the local coordinate system
    # ============================================
    # Call appropriate coordinate system functions based on stiffness flag
    @ti.kernel
    def get_input2(self):
        # Loop over nodes to initialize s_xm0 and s_q_fg0
        for ip in range(0, var.s_nnode[None]):
            var.s_xm0[ip] = var.s_xm[ip]
            var.s_q_fg0[ip][0, :] = var.s_e_fi1[ip]
            var.s_q_fg0[ip][1, :] = var.s_e_fi2[ip]
            var.s_q_fg0[ip][2, :] = var.s_e_fi3[ip]
        # Loop over Gauss points to initialize s_xmk0 and s_q_gl0
        for ik in range(0, var.s_mgk[None]):
            var.s_xmk0[ik] = var.s_xmk[ik]
            var.s_q_gl0[ik][0, :] = var.s_e_la1[ik]
            var.s_q_gl0[ik][1, :] = var.s_e_la2[ik]
            var.s_q_gl0[ik][2, :] = var.s_e_la3[ik]
        # Output the status of the local coordinate system establishment
        print('*** Shell local coordinate system has established !')

        # !c==========================================c
        # !c==========================================c
        # !c...赋不同的板厚
        # !c
        # Assign different thicknesses for nodes
        for ip in range(0, var.s_nnode[None]):
            var.s_thk[ip] = 0.01
            var.s_dvom[ip] = var.s_darea[ip] * var.s_thk[ip]
        
        # Assign different thicknesses for Gauss points
        for ik in range(0, var.s_mgk[None]):
            var.s_thkk[ik] = 0.01
            var.s_dvomk[ik] = var.s_dareak[ik] * var.s_thkk[ik]
        
        # Assign different mat types for nodes
        for ip in range(0, var.s_nnode[None]):
            mat.s_MatType[ip] = 0
        
        # Assign different mat types for Gauss points
        for ik in range(0, var.s_mgk[None]):
            mat.s_MatTypek[ik] = 0

    @ti.kernel
    def get_input3(self):
        # !c==========================================c
        # !c  Calculate: Mass matrix & Inertia        c
        # !c             Lumped nodal volume          c
        # !c==========================================c
        # !c
        eps = 0.0
        var.dvtemp[None] = 0.0  # The volume of the whole model
        var.s_mass.fill(0.0)
        var.s_mass_1.fill(0.0)
        alpha = 0.0
        alpha0 = 0.0  # sum of diagonal terms of consistent mass matrix
        alpha1 = 0.0  # total mass

        s_Lumping = params.s_Lumping[None]
        
        # Lumped mass matrix calculation
        if s_Lumping == 1:

            for ik in range(0, var.s_mgk[None]):
                jLoop = params.s_maxNode[None]
                for jp in range(0, jLoop):
                    alpha0 += mat.el_rho[mat.s_MatTypek[ik]] * var.s_dvomk[ik] * var.s_shpk[jp, ik] ** 2
                    alpha1 += mat.el_rho[mat.s_MatTypek[ik]] * var.s_dvomk[ik] * var.s_shpk[jp, ik]
                    var.dvtemp[None] += var.s_dvomk[ik] * var.s_shpk[jp, ik]
            
            alpha = alpha1 / alpha0
        
        # Loop to update mass for each node and Gauss point
        for ik in range(0, var.s_mgk[None]):
            jLoop = params.s_maxNode[None]
            for jp in range(1, jLoop + 1):
                jpt = var.s_nele_node[jp, ik]
                
                if s_Lumping == 0:
                    var.s_mass[jpt-1] += mat.el_rho[mat.s_MatTypek[ik]] * var.s_dvomk[ik] * var.s_shpk[jp-1, ik]
                    if var.s_dvomk[ik] <= eps:
                        print(f"*** Wrong s_dvomk of node: {ik+1}")
                        print(f"*** The value of s_dvomk is: {var.s_dvomk[ik]}")
                elif s_Lumping == 1:
                    # Hinton lump technique(Hughes>>FEM-LS&DFEA,page:445)
                    var.s_mass[jpt-1] += alpha * mat.el_rho[mat.s_MatTypek[ik]] * var.s_dvomk[ik] * var.s_shpk[jp-1, ik] ** 2
        
        # Output the mass and dvtemp
        print('*** Mass is done!')
        
        #!c
        #!c..Lumped nodal Inertia (see hughes etc. 2009 isogeometry analysis of shell)
        #!c
        for ipt in range(0, var.s_nnode[None]):
            var.s_inertia[ipt] = var.s_mass[ipt] * (var.s_thk[ipt] ** 2 / 12.0 + var.s_darea[ipt] / 12.0)
        
        # Total mass calculation
        var.s_tom[None] = 0.0
        for ipt in range(0, var.s_nnode[None]):
            var.s_tom[None] += var.s_mass[ipt]
            if var.s_mass[ipt] <= eps:
                print(f"{ipt+1} Singularity in M")
            
            var.s_mass_1[ipt] = 1.0 / var.s_mass[ipt]
            var.s_inertia_1[ipt] = 1.0 / var.s_inertia[ipt]
        var.dvtemp[None] = var.s_tom[None]/mat.el_rho[mat.s_MatTypek[0]]
        # Output total mass
        print(f"*** dvtemp = {var.dvtemp[None]:12.4e}")
        print(f"*** Total mass is : {var.s_tom[None]:12.4e}")

