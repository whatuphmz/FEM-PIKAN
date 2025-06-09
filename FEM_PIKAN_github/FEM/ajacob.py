import numpy as np
import torch
import taichi as ti

# !c=========================================================c
# !c   cal the determinant of J matrix for 4-node element    c
# !c                                                         c
# !c              |      i          j           k      |     c
# !c              |                                    |     c
# !c              |      dx         dy          dz     |     c
# !c     det:     |    ______      ______      ______  |     c
# !c              |     dxsi        dxsi        dxsi   |     c
# !c              |                                    |     c
# !c              |      dx         dy          dz     |     c
# !c              |    ______      ______      ______  |     c
# !c              |     deta        deta        deta   |     c
# !c=========================================================c
# !c
@ti.func
def ajacob4_space(x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, xsi, eta):

    # !...求xmk(1,ip)=shape(1)*xn1+shape(2)*xn2+shape(3)*xn3+shape(4)*xn4对xsi的导数,只需求shape对xsi的导数即可
    # !...求gauss积分在局部坐标系下求完后要成以这个Jacobi行列式【stokes定理】

    aj11 = (1. - eta) * (x2 - x1) / 4. + (1. + eta) * (x3 - x4) / 4.
    aj12 = (1. - eta) * (y2 - y1) / 4. + (1. + eta) * (y3 - y4) / 4.
    aj13 = (1. - eta) * (z2 - z1) / 4. + (1. + eta) * (z3 - z4) / 4.
    aj21 = (1. - xsi) * (x4 - x1) / 4. + (1. + xsi) * (x3 - x2) / 4.
    aj22 = (1. - xsi) * (y4 - y1) / 4. + (1. + xsi) * (y3 - y2) / 4.
    aj23 = (1. - xsi) * (z4 - z1) / 4. + (1. + xsi) * (z3 - z2) / 4.
    
    # Calculate the determinant of the Jacobian matrix
    ajj = ti.sqrt((aj11 * aj22 - aj12 * aj21) ** 2 +
                  (aj12 * aj23 - aj22 * aj13) ** 2 +
                  (aj13 * aj21 - aj23 * aj11) ** 2)
    
    return ajj

# !c=========================================================c
# !c  cal the determinant of J matrix for 4-node element     c
# !c                                                         c
# !c                 |      dx         dx       |            c
# !c                 |    ______      ______    |            c
# !c                 |     dxsi        deta     |            c
# !c            det  |                          |            c
# !c                 |      dy         dy       |            c
# !c                 |    ______      ______    |            c
# !c                 |     dxsi        deta     |            c
# !c=========================================================c
# !c
@ti.func
def ajacob4_plate(x1, x2, x3, x4, y1, y2, y3, y4, xai, eta):
    """
    Calculate the determinant of the J matrix for a 4-node element in 2D (plate).
    """

    # Compute partial derivatives
    aj11 = (1. - eta) * (x2 - x1) / 4. + (1. + eta) * (x3 - x4) / 4.
    aj12 = (1. - xai) * (x4 - x1) / 4. + (1. + xai) * (x3 - x2) / 4.
    aj21 = (1. - eta) * (y2 - y1) / 4. + (1. + eta) * (y3 - y4) / 4.
    aj22 = (1. - xai) * (y4 - y1) / 4. + (1. + xai) * (y3 - y2) / 4.
    
    # Calculate the determinant of the Jacobian matrix
    ajj = aj11 * aj22 - aj12 * aj21
    
    return ajj

    # x1, x2, x3, x4 = X[:,0], X[:,1], X[:,2], X[:,3]
    # y1, y2, y3, y4 = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    # z1, z2, z3, z4 = Z[:,0], Z[:,1], Z[:,2], Z[:,3]
