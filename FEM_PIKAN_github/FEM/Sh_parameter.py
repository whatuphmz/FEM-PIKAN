import taichi as ti


@ti.data_oriented
class Sh_parameter:
    def __init__(self):

        self.epochs = ti.field(ti.i32, shape=())
        self.opt_switch_epoch = ti.field(ti.i32, shape=())

        # Set print and save frequency
        self.s_interact_stat = ti.field(ti.i32, shape=())
        self.s_nHoutFrq = ti.field(ti.i32, shape=())
        self.s_nprintFrq = ti.field(ti.i32, shape=())
        self.s_nPlotFrq = ti.field(ti.i32, shape=())

        # Integration method, time increment, and PI
        self.gamma = ti.field(ti.f32, shape=())
        self.beta = ti.field(ti.f32, shape=())
        self.theta = ti.field(ti.f32, shape=())
        self.s_iIntMeth = ti.field(ti.i32, shape=())
        self.s_interp_meth = ti.field(ti.i32, shape=())
        self.s_iInter = ti.field(ti.i32, shape=())
        self.s_pi = ti.field(ti.f32, shape=())

        # External force type
        self.s_body_force = ti.field(ti.i32, shape=())
        self.s_edge_load = ti.field(ti.i32, shape=())
        self.s_surface_pressure = ti.field(ti.i32, shape=())
        self.s_LoadingType = ti.field(ti.i32, shape=())
        self.s_bulk_viscosity_sign = ti.field(ti.i32, shape=())
        self.s_hourglass_control = ti.field(ti.i32, shape=())

        # Stability and other algorithm settings
        self.s_contact = ti.field(ti.i32, shape=())
        self.s_stiffness = ti.field(ti.i32, shape=())
        self.s_with_beam = ti.field(ti.i32, shape=())
        self.s_ini_crack = ti.field(ti.i32, shape=())
        self.s_crack_grow = ti.field(ti.i32, shape=())

        # Computational geometry parameters
        self.s_x_maxgeom = ti.field(ti.f32, shape=())
        self.s_x_mingeom = ti.field(ti.f32, shape=())
        self.s_y_maxgeom = ti.field(ti.f32, shape=())
        self.s_y_mingeom = ti.field(ti.f32, shape=())
        self.s_z_maxgeom = ti.field(ti.f32, shape=())
        self.s_z_mingeom = ti.field(ti.f32, shape=())
        self.maxngx = ti.field(ti.i32, shape=())
        self.maxngy = ti.field(ti.i32, shape=())
        self.maxngz = ti.field(ti.i32, shape=())

        # Dimension of the problem and search method
        self.s_ndim = ti.field(ti.i32, shape=())
        self.s_search_method = ti.field(ti.i32, shape=())
        self.s_wft = ti.field(ti.i32, shape=())

        # Control output
        self.s_output_format = ti.field(ti.i32, shape=())
        self.s_out_Binary = ti.field(ti.i32, shape=())
        self.s_geometry_node = ti.field(ti.i32, shape=())
        self.s_geometry_ele = ti.field(ti.i32, shape=())
        self.s_disp_nodal = ti.field(ti.i32, shape=())
        self.s_damage_nodal = ti.field(ti.i32, shape=())
        self.s_nodal_type = ti.field(ti.i32, shape=())
        self.s_nodal_normal = ti.field(ti.i32, shape=())
        self.s_pressure_node_out = ti.field(ti.i32, shape=())
        self.s_velocity_node_out = ti.field(ti.i32, shape=())
        self.s_pressure_ele_out = ti.field(ti.i32, shape=())
        self.s_velocity_ele_out = ti.field(ti.i32, shape=())
        self.s_Mises_out = ti.field(ti.i32, shape=())
        self.s_PEEQ_out = ti.field(ti.i32, shape=())
        self.s_mass_output = ti.field(ti.i32, shape=())
        self.s_Istage = ti.field(ti.i32, shape=())
        self.s_shp_info_out = ti.field(ti.i32, shape=())
        self.s_Gauss_kinematic = ti.field(ti.i32, shape=())
        self.s_force_deflection = ti.field(ti.i32, shape=())

        # Lumping and epsilon values
        self.s_Lumping = ti.field(ti.i32, shape=())            # 0 for row sum technique, 1 for hinton lump technique
        self.teps = ti.field(ti.f32, shape=())

        # Adaptive algorithm parameters
        self.s_maxMat = ti.field(ti.i32, shape=())
        self.s_maxElem = ti.field(ti.i32, shape=())
        self.s_maxNumnp = ti.field(ti.i32, shape=())
        self.s_maxNode = ti.field(ti.i32, shape=())
        self.s_maxInt = ti.field(ti.i32, shape=())             # Choose 1 for reduced integration
        self.s_maxntg = ti.field(ti.i32, shape=())
        self.s_maxIntElem = ti.field(ti.i32, shape=())
        self.s_maxGP = ti.field(ti.i32, shape=())
        self.s_max_contact_node = ti.field(ti.i32, shape=())  # 10 + 8 + 8

        # Beam and 1D element parameters
        self.maxnode1D = ti.field(ti.i32, shape=())
        self.maxBeamNump = ti.field(ti.i32, shape=())
        self.maxBeamElem = ti.field(ti.i32, shape=())
        self.beam_mnsch = ti.field(ti.i32, shape=())
        self.maxBeamGP = ti.field(ti.i32, shape=())

        # Boundary condition limits
        self.s_maxDispbcX = ti.field(ti.i32, shape=())
        self.s_maxDispbcY = ti.field(ti.i32, shape=())
        self.s_maxDispbcZ = ti.field(ti.i32, shape=())
        self.s_maxVelbcX = ti.field(ti.i32, shape=())
        self.s_maxVelbcY = ti.field(ti.i32, shape=())
        self.s_maxVelbcZ = ti.field(ti.i32, shape=())
        self.s_maxEssbcX = ti.field(ti.i32, shape=())
        self.s_maxEssbcY = ti.field(ti.i32, shape=())
        self.s_maxEssbcZ = ti.field(ti.i32, shape=())
        self.s_maxAngDispbcX = ti.field(ti.i32, shape=())
        self.s_maxAngDispbcY = ti.field(ti.i32, shape=())
        self.s_maxAngDispbcZ = ti.field(ti.i32, shape=())
        self.s_maxAngVelbcX = ti.field(ti.i32, shape=())
        self.s_maxAngVelbcY = ti.field(ti.i32, shape=())
        self.s_maxAngVelbcZ = ti.field(ti.i32, shape=())
        self.s_maxAngEssbcX = ti.field(ti.i32, shape=())
        self.s_maxAngEssbcY = ti.field(ti.i32, shape=())
        self.s_maxAngEssbcZ = ti.field(ti.i32, shape=())
        self.s_maxEssbc = ti.field(ti.i32, shape=())

        # Particle support parameters
        self.s_mnsch = ti.field(ti.i32, shape=())
        self.s_maxNtip = ti.field(ti.i32, shape=())        # Max number of crack tips
        self.s_maxCrack = ti.field(ti.i32, shape=())       # Max number of crack paths

        self.epochs[None] = 10000
        self.opt_switch_epoch[None] = 0

        # Set print and save frequency
        self.s_interact_stat[None] = True
        self.s_nHoutFrq[None] = 4           # Field output frequency
        self.s_nprintFrq[None] = 100        # Print to screen frequency
        self.s_nPlotFrq[None] = 4000        # Output to file frequency

        # Integration method, time increment, and PI
        self.gamma[None] = 0.5
        self.beta[None] = 0.0
        self.theta[None] = 0.5
        self.s_iIntMeth[None] = 1           # Choose 3 for rate-constitutive model; 4 or 5 for quasi-static simulation
        self.s_interp_meth[None] = 1        # 0 for FEM, 1 for RKPM
        self.s_iInter[None] = 1             # 1 for bilinear interpolation, 2 for poly interpolation
        self.s_pi[None] = 3.141592653589793

        # External force type
        self.s_body_force[None] = 1
        self.s_edge_load[None] = 1
        self.s_surface_pressure[None] = 1
        self.s_LoadingType[None] = 0        # Type of load vs. time
        self.s_bulk_viscosity_sign[None] = 0
        self.s_hourglass_control[None] = False

        # Stability and other algorithm settings
        self.s_contact[None] = False
        self.s_stiffness[None] = 1
        self.s_with_beam[None] = False
        self.s_ini_crack[None] = False
        self.s_crack_grow[None] = False

        # Computational geometry parameters
        self.s_x_maxgeom[None] = 0.1 + 0.02
        self.s_x_mingeom[None] = -0.1 - 0.02
        self.s_y_maxgeom[None] = 0.2 + 0.02
        self.s_y_mingeom[None] = -0.0 - 0.02
        self.s_z_maxgeom[None] = 0.3 + 0.02
        self.s_z_mingeom[None] = 0.0 - 0.02
        self.maxngx[None] = 160
        self.maxngy[None] = 160
        self.maxngz[None] = 320

        # Dimension of the problem and search method
        self.s_ndim[None] = 3
        self.s_search_method[None] = 2
        self.s_wft[None] = 1                 # Window function type: 1 for cubic spline

        # Control output
        self.s_output_format[None] = 2
        self.s_out_Binary[None] = False
        self.s_geometry_node[None] = False
        self.s_geometry_ele[None] = True
        self.s_disp_nodal[None] = True
        self.s_damage_nodal[None] = True
        self.s_nodal_type[None] = True
        self.s_nodal_normal[None] = True
        self.s_pressure_node_out[None] = False
        self.s_velocity_node_out[None] = True
        self.s_pressure_ele_out[None] = False
        self.s_velocity_ele_out[None] = True
        self.s_Mises_out[None] = 1
        self.s_PEEQ_out[None] = 1
        self.s_mass_output[None] = 1
        self.s_Istage[None] = 1
        self.s_shp_info_out[None] = True
        self.s_Gauss_kinematic[None] = True
        self.s_force_deflection[None] = 0

        # File identifiers
        self.s_ifrInp = 101
        self.s_ifwEcho = 102
        self.s_ifwMass = 103
        self.s_ifwout = 104
        self.s_ifwfdcurve = 105
        self.s_ifwLoad = 106
        self.s_ifwVelLoad = 107
        self.s_ifwError = 108
        self.s_ifwCrack = 109
        self.s_ifwHoutput = 110
        self.s_shape_out = 111
        self.s_shapek_out = 112
        self.s_shapen_out = 113
        self.s_ifwoutvtk = 114
        self.FEM_result = 115
        self.FEM_result_out = 116
        self.si_ifrInp = 117
        self.f_ifwfdcurve = 118

        # Lumping and epsilon values
        self.s_Lumping[None] = 0            # 0 for row sum technique, 1 for hinton lump technique
        self.teps[None] = 1.99999999999

        # Adaptive algorithm parameters
        self.s_maxMat[None] = 30
        self.s_maxElem[None] = 100000
        self.s_maxNumnp[None] = 105000
        self.s_maxNode[None] = 4
        self.s_maxInt[None] = 1             # Choose 1 for reduced integration
        self.s_maxntg[None] = 3
        self.s_maxIntElem[None] = self.s_maxInt[None] * self.s_maxInt[None]
        self.s_maxGP[None] = self.s_maxElem[None] * self.s_maxIntElem[None]
        self.s_max_contact_node[None] = 26  # 10 + 8 + 8

        # Beam and 1D element parameters
        self.maxnode1D[None] = 2
        self.maxBeamNump[None] = 200
        self.maxBeamElem[None] = 200
        self.maxBeamGP[None] = self.maxBeamElem[None] * self.s_maxInt[None]
        self.beam_mnsch[None] = 20

        # Boundary condition limits
        self.s_maxDispbcX[None] = 2000
        self.s_maxDispbcY[None] = 2000
        self.s_maxDispbcZ[None] = 2000
        self.s_maxVelbcX[None] = 2000
        self.s_maxVelbcY[None] = 2000
        self.s_maxVelbcZ[None] = 2000
        self.s_maxEssbcX[None] = self.s_maxDispbcX[None] + self.s_maxVelbcX[None]
        self.s_maxEssbcY[None] = self.s_maxDispbcY[None] + self.s_maxVelbcY[None]
        self.s_maxEssbcZ[None] = self.s_maxDispbcZ[None] + self.s_maxVelbcZ[None]
        self.s_maxAngDispbcX[None] = 2000
        self.s_maxAngDispbcY[None] = 2000
        self.s_maxAngDispbcZ[None] = 2000
        self.s_maxAngVelbcX[None] = 2000
        self.s_maxAngVelbcY[None] = 2000
        self.s_maxAngVelbcZ[None] = 2000
        self.s_maxAngEssbcX[None] = self.s_maxAngDispbcX[None] + self.s_maxAngVelbcX[None]
        self.s_maxAngEssbcY[None] = self.s_maxAngDispbcY[None] + self.s_maxAngVelbcY[None]
        self.s_maxAngEssbcZ[None] = self.s_maxAngDispbcZ[None] + self.s_maxAngVelbcZ[None]
        self.s_maxEssbc[None] = self.s_maxEssbcX[None] + self.s_maxEssbcY[None] + self.s_maxEssbcZ[None]

        # Particle support parameters
        self.s_mnsch[None] = 50
        self.s_maxNtip[None] = 5000         # Max number of crack tips
        self.s_maxCrack[None] = 20000       # Max number of crack paths

# Instantiate the ShParameter class
params = Sh_parameter()