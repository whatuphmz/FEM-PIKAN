import numpy as np
import torch
import taichi as ti

@ti.func
def evl_FEM_shape4(xsi, eta):

    # !c======================================================c
    # !c...this subroutine is to evaluate:                    c
    # !c       the shape function values for 4-node element   c
    # !c                                                      c
    # !c			    y                                     c
    # !c	            ^                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c        4---------------3                             c
    # !c        |		|       |                             c
    # !c        |  	    |       |                             c
    # !c        | 		|       |                             c
    # !c	    |		|-------|--------------------> x      c
    # !c	    |		        |                             c
    # !c	    |  	            |                             c
    # !c	    |		        |                             c
    # !c		1---------------2                             c
    # !c                                                      c
    # !c======================================================c
    # !c
    shapef = ti.Vector.zero(ti.f32, 4)
    
    shapef[0] = 0.25 * (1. - xsi) * (1. - eta)
    shapef[1] = 0.25 * (1. + xsi) * (1. - eta)
    shapef[2] = 0.25 * (1. + xsi) * (1. + eta)
    shapef[3] = 0.25 * (1. - xsi) * (1. + eta)

    return shapef

@ti.func
def evl_FEM_dshape4(xsi, eta):

    # !c======================================================c
    # !c  this subroutine is to evaluate:                     c
    # !c    derivatives of shape function for 4-node element  c 
    # !c                                                      c
    # !c			    y                                     c	
    # !c	            ^                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c			    |                                     c
    # !c        4---------------3                             c
    # !c        |		|       |                             c
    # !c        |  	    |       |                             c
    # !c        | 		|       |                             c
    # !c	    |		|-------|--------------------> x      c
    # !c	    |		        |                             c
    # !c	    |  	            |                             c
    # !c	    |		        |                             c
    # !c		1---------------2                             c
    # !c                                                      c
    # !c======================================================c
    # !c
    dshape_xsi = ti.Vector.zero(ti.f32, 4)
    dshape_eta = ti.Vector.zero(ti.f32, 4)
    
    dshape_xsi[0] = -0.25 * (1. - eta)
    dshape_xsi[1] =  0.25 * (1. - eta)
    dshape_xsi[2] =  0.25 * (1. + eta)
    dshape_xsi[3] = -0.25 * (1. + eta)
    
    dshape_eta[0] = -0.25 * (1. - xsi)
    dshape_eta[1] = -0.25 * (1. + xsi)
    dshape_eta[2] =  0.25 * (1. + xsi)
    dshape_eta[3] =  0.25 * (1. - xsi)
    
    return dshape_xsi, dshape_eta