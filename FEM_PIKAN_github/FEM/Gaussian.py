import taichi as ti
from FEM.Sh_cache import var



@ti.func
def gauss1D(nint):

    gp_loc1D = ti.Vector([0.0] * 10)
    gp_weight1D = ti.Vector([0.0] * 10)
    
    assert nint <= 10, f"Unsupported nint={nint} in gauss1D."
    
    if nint == 1:
        gp_loc1D[0] = 0.0
        gp_weight1D[0] = 2.0
    elif nint == 2:
        gp_loc1D[0] = -0.577350269189626
        gp_loc1D[1] = -gp_loc1D[0]
        for i in range(nint):
            gp_weight1D[i] = 1.0
    elif nint == 3:
        gp_loc1D[0] = -0.774596669241483
        gp_loc1D[1] = 0.0
        gp_loc1D[2] = -gp_loc1D[0]
        gp_weight1D[0] = gp_weight1D[2] = 0.555555555555556
        gp_weight1D[1] = 0.888888888888889
    elif nint == 4:
        gp_loc1D[0] = -0.861136311594053
        gp_loc1D[1] = -0.339981043584856
        gp_loc1D[2] = -gp_loc1D[1]
        gp_loc1D[3] = -gp_loc1D[0]
        gp_weight1D[0] = gp_weight1D[3] = 0.347854845137454
        gp_weight1D[1] = gp_weight1D[2] = 0.652145154862546
    elif nint == 5:
        gp_loc1D[0] = -0.906179845938664
        gp_loc1D[1] = -0.538469310105683
        gp_loc1D[2] = 0.0
        gp_loc1D[3] = -gp_loc1D[1]
        gp_loc1D[4] = -gp_loc1D[0]
        gp_weight1D[0] = gp_weight1D[4] = 0.236936885056189
        gp_weight1D[1] = gp_weight1D[3] = 0.478638670499366
        gp_weight1D[2] = 0.568888888888889
    else:
        assert nint <= 5, f"Unsupported nint={nint} in gauss1D."

    return gp_loc1D, gp_weight1D


# 定义2D Gauss积分函数
@ti.func
def gauss2D(s_ele_node, s_nintElem):
    # !c=========================================================c
    # !c...this subroutine to get the loc coordinates and        c
    # !c        weights for the 2D elements                      c
    # !c                                                         c
    # !c...input parameter:                                      c
    # !c        nintElem: the total number of integration        c
    # !c                  points per element                     c
    # !c        elenode : the node number of each element        c
    # !c            elenode = 3: triangle element                c
    # !c                      4: quadrilateral element           c
    # !c=========================================================c
    # !c
    if s_ele_node[None] == 4:  # Quadrilateral element
        nint1D = int(ti.sqrt(s_nintElem[None]))
        gp_loc1D, gp_weight1D = gauss1D(nint1D)

        iLocInt = 0
        for ig in range(nint1D):
            xsi = gp_loc1D[ig]
            for jg in range(nint1D):
                eta = gp_loc1D[jg]
                var.gp_loc2D[0, iLocInt] = xsi
                var.gp_loc2D[1, iLocInt] = eta
                var.gp_weight2D[iLocInt] = gp_weight1D[ig] * gp_weight1D[jg]
                iLocInt += 1

    elif s_ele_node[None] == 3:  # Triangle element

        if s_nintElem[None] == 1:
            var.gp_loc2D[0, 0] = 1/3
            var.gp_loc2D[1, 0] = 1/3
            var.gp_weight2D[0] = 1.0
        elif s_nintElem[None] == 3:
            var.gp_loc2D[0, 0] = 0.5
            var.gp_loc2D[1, 0] = 0.5
            var.gp_loc2D[0, 1] = 0.0
            var.gp_loc2D[1, 1] = 0.5
            var.gp_loc2D[0, 2] = 0.5
            var.gp_loc2D[1, 2] = 0.0
            var.gp_weight2D[0] = 1/3
            var.gp_weight2D[1] = 1/3
            var.gp_weight2D[2] = 1/3
        elif s_nintElem[None] == 4:
            var.gp_loc2D[0, 0] = 1/3
            var.gp_loc2D[1, 0] = 1/3
            var.gp_weight2D[0] = -27/48
            var.gp_loc2D[0, 1] = 0.6
            var.gp_loc2D[1, 1] = 0.2
            var.gp_weight2D[1] = 25/48
            var.gp_loc2D[0, 2] = 0.2
            var.gp_loc2D[1, 2] = 0.6
            var.gp_weight2D[2] = 25/48
            var.gp_loc2D[0, 3] = 0.2
            var.gp_loc2D[1, 3] = 0.2
            var.gp_weight2D[3] = 25/48
        else:
            print(f"Error: Unknown s_nintElem={s_nintElem[None]} for triangle element.")

    else:
        print(f"Error: Unknown s_ele_node={s_ele_node[None]}.")
