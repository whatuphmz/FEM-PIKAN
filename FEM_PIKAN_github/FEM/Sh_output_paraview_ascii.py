import numpy as np
from FEM.Sh_cache import var
from FEM.Sh_parameter import params
import taichi as ti

@ti.data_oriented
class Sh_output_paraview_ascii:
    def __init__(self):
        ctemp = f"{var.nframe[None]:04d}"
        file_name = f"data/timeSh_{ctemp}.vtu"
        
        with open(file_name, 'w') as f:
            # ================== XML 头部 ==================
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('<UnstructuredGrid>\n')
            
            # ================== Piece定义 ==================
            f.write(f'<Piece NumberOfPoints="{var.s_nnode[None]}" NumberOfCells="{var.s_nelement[None]}">\n')
            
            # ================== 节点坐标 ==================
            f.write('<Points>\n')
            f.write('<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            points_data = "\n".join([f"{var.s_xm0[ip][0]:.5e} {var.s_xm0[ip][1]:.5e} {var.s_xm0[ip][2]:.5e}" 
                                   for ip in range(var.s_nnode[None])])
            f.write(points_data + '\n')
            f.write('</DataArray>\n</Points>\n')
            
            # ================== 单元定义 ==================
            f.write('<Cells>\n')
            # 单元连接性
            f.write('<DataArray Name="connectivity" type="Int32" format="ascii">\n')
            conn_data = "\n".join([f"{var.s_nele_node[1, ip]-1} {var.s_nele_node[2, ip]-1} "
                                  f"{var.s_nele_node[3, ip]-1} {var.s_nele_node[4, ip]-1}" 
                                  for ip in range(var.s_nelement[None])])
            f.write(conn_data + '\n</DataArray>\n')
            
            # 单元偏移量（四边形每个单元偏移4个节点）
            f.write('<DataArray Name="offsets" type="Int32" format="ascii">\n')
            offsets = [4*(i+1) for i in range(var.s_nelement[None])]
            f.write("\n".join(map(str, offsets)) + '\n</DataArray>\n')
            
            # 单元类型（四边形对应类型码9）
            f.write('<DataArray Name="types" type="UInt8" format="ascii">\n')
            types_data = "\n".join(["9" for _ in range(var.s_nelement[None])])
            f.write(types_data + '\n</DataArray>\n</Cells>\n')
            
            # ================== 节点数据 ==================
            f.write('<PointData>\n')
            
            # Mises应力（标量）
            f.write('<DataArray Name="Mises" type="Float32" format="ascii">\n')
            stress_data = "\n".join([f"{var.s_effstsnp[ip]:.5e}" for ip in range(var.s_nnode[None])])
            f.write(stress_data + '\n</DataArray>\n')
            
            # 位移向量
            def write_vector(name, data_field):
                f.write(f'<DataArray Name="{name}" type="Float32" NumberOfComponents="3" format="ascii">\n')
                vector_data = "\n".join([f"{data_field[ip][0]:.5e} {data_field[ip][1]:.5e} {data_field[ip][2]:.5e}" 
                                       for ip in range(var.s_nnode[None])])
                f.write(vector_data + '\n</DataArray>\n')
            
            write_vector("Disp", var.s_disp)
            write_vector("angDisp", var.s_angdisp)
            write_vector("Fiber", var.s_e_fi3)
            write_vector("Disp_FEM", var.disp_FEM)
            write_vector("angDisp_FEM", var.angdisp_FEM)
            write_vector("Disp_error", var.disp_error)
            write_vector("angDisp_error", var.angdisp_error)
            
            f.write('</PointData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>')
            
        print(f"VTU data written to {file_name}")