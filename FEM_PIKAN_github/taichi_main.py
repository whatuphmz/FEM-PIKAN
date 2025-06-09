import taichi as ti
ti.init(arch=ti.gpu)
import os
import matplotlib.pyplot as plt
from taichi_optimizer_firstorder import *
from Networks import *
from FEM.Sh_cache import var
from FEM.Sh_parameter import params
from FEM.Sh_input import Sh_input
from FEM.Sh_coordinate_system_stiff import *
from FEM.Sh_update_fiber_quater import *
from FEM.Sh_K_element import *
from FEM.Sh_external_force import *
from FEM.Sh_output_paraview_ascii import *
from HRKAN import *
import pandas as pd
import math


TRAIN = True

@ti.kernel
def import_data(data: ti.types.ndarray()):
    var.disp_FEM_mean[None] = 0.0
    var.angdisp_FEM_mean[None] = 0.0
    for ip in range(var.s_nnode[None]):
        if var.s_ibtype[ip] != 200:
            for i in ti.static(range(3)):
                var.disp_FEM[ip][i] = data[ip, i+1]
                var.angdisp_FEM[ip][i] = data[ip, i+4]
            var.disp_FEM_mean[None] += var.disp_FEM[ip].dot(var.disp_FEM[ip])
            var.angdisp_FEM_mean[None] += var.angdisp_FEM[ip].dot(var.angdisp_FEM[ip])
    var.disp_FEM_mean[None] /= (3 * var.s_nfree[None])
    var.angdisp_FEM_mean[None] /= (3 * var.s_nfree[None])

def init_nn_model():
    global model_num, BATCH_SIZE, steps, n_output_act, input_states, fc1, fc2, fc3#, fc4, fc5
    global loss, Disp, Angdisp, Wint, Wext
    global optimizer
    # NN model
    model_num = 1
    steps = 6
    n_input = 3
    n_hidden = 32
    n_output_act = 1
    learning_rate = 1e-5
    BATCH_SIZE = var.s_nnode[None]
    loss = ti.field(float, shape=(), needs_grad=True)
    Wint = ti.field(float, shape=(var.s_nnode[None]), needs_grad=True)
    Wext = ti.field(float, shape=(var.s_nnode[None]), needs_grad=True)

    if TRAIN:
        Disp = ti.Vector.field(3, ti.f32, shape=(var.s_nnode[None]), needs_grad=True)
        Angdisp = ti.Vector.field(3, ti.f32, shape=(var.s_nnode[None]), needs_grad=True)
        input_states = ti.field(float, shape=(model_num, steps, BATCH_SIZE, n_input), needs_grad=True)
        fc1 = HRKAN_input(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_input,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=False,
        )
        fc2 = HRKAN(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=False,
        )
        # fc3 = Linear(
        #     n_models=model_num,
        #     batch_size=BATCH_SIZE,
        #     n_steps=steps,
        #     n_input=n_hidden,
        #     n_hidden=n_hidden,
        #     n_output=n_hidden,
        #     needs_grad=True,
        #     activation=True,
        # )
        # fc4 = Linear(
        #     n_models=model_num,
        #     batch_size=BATCH_SIZE,
        #     n_steps=steps,
        #     n_input=n_hidden,
        #     n_hidden=n_hidden,
        #     n_output=n_hidden,
        #     needs_grad=True,
        #     activation=True,
        # )
        # fc5 = Linear(
        #     n_models=model_num,
        #     batch_size=BATCH_SIZE,
        #     n_steps=steps,
        #     n_input=n_hidden,
        #     n_hidden=n_hidden,
        #     n_output=n_hidden,
        #     needs_grad=True,
        #     activation=True,
        # )
        fc3 = HRKAN_output(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_output_act,
            n_output=n_output_act,
            needs_grad=True,
            activation=False,
        )
        fc1.weights_init()
        fc2.weights_init()
        fc3.weights_init()
        # fc4.weights_init()
        # fc5.weights_init()
        NNs = [fc1, fc2, fc3]#, fc4, fc5
        parameters = []
        for layer in NNs:
            parameters.extend(layer.parameters())
        optimizer = AMSGrad(params=parameters, lr=learning_rate)
    else:
        input_states = ti.field(float, shape=(model_num, steps, BATCH_SIZE, n_input), needs_grad=False)
        fc1 = Linear(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_input,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=True,
        )
        fc2 = Linear(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=True,
        )
        fc3 = Linear(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=True,
        )
        fc4 = Linear(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_hidden,
            n_output=n_hidden,
            needs_grad=True,
            activation=True,
        )
        fc5 = Linear_output(
            n_models=model_num,
            batch_size=BATCH_SIZE,
            n_steps=steps,
            n_input=n_hidden,
            n_hidden=n_output_act,
            n_output=n_output_act,
            needs_grad=True,
            activation=True,
        )
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        fc1.load_weights(f"{file_dir_path}/saved_models/plate_-1000g/fc1_399999.pkl", model_id=0)
        fc2.load_weights(f"{file_dir_path}/saved_models/plate_-1000g/fc2_399999.pkl", model_id=0)
        fc3.load_weights(f"{file_dir_path}/saved_models/plate_-1000g/fc3_399999.pkl", model_id=0)
        fc4.load_weights(f"{file_dir_path}/saved_models/plate_-1000g/fc4_399999.pkl", model_id=0)
        fc5.load_weights(f"{file_dir_path}/saved_models/plate_-1000g/fc5_399999.pkl", model_id=0)
        print(f"Model at {file_dir_path} loaded. ")

@ti.kernel
def fill_input_states(current_pos: ti.int32):
    b = ti.Vector([0.5, 0.5, 1.0])
    for t, bs in ti.ndrange(steps, (current_pos, current_pos + BATCH_SIZE)):
        for j in ti.static(range(3)):
            input_states[0, t, bs-current_pos, j] = var.s_xm0[bs][j] / b[j]

a = ti.Vector([1e-8, 1e-8, 1, 1e1, 1e1, 1e-8])

@ti.kernel
def update_PINN(t: ti.int32, current_pos: ti.int32):
    for ip in range(current_pos, current_pos + BATCH_SIZE):
        if t < 3:
            Disp[ip][t] = fc3.output[0, t, ip-current_pos, 0]*a[t]
        else:
            Angdisp[ip][t-3] = fc3.output[0, t, ip-current_pos, 0]*a[t]

@ti.kernel
def compute_loss(current_pos: ti.int32):
    for ip in range(var.s_nnode[None]):
        if current_pos <= ip < current_pos + BATCH_SIZE:
            var.Fsint[ip].fill(0.0)
            var.Msint[ip].fill(0.0)
    for ik in range(var.s_mgk[None]):
        for iz in range(params.s_maxNode[None]):
            ip = var.s_nele_node[iz + 1, ik] - 1
            if current_pos <= ip < current_pos + BATCH_SIZE:
                for jz in range(params.s_maxNode[None]):
                    jp = var.s_nele_node[jz + 1, ik] - 1
                    if ip == jp:
                        var.Fsint[ip] += var.s_Kvv[iz, jz, ik] @ Disp[jp] + var.s_Kvw[iz, jz, ik] @ Angdisp[jp]
                        var.Msint[ip] += var.s_Kwv[iz, jz, ik] @ Disp[jp] + var.s_Kww[iz, jz, ik] @ Angdisp[jp]
                    else:
                        var.Fsint[ip] += var.s_Kvv[iz, jz, ik] @ var.s_disp[jp] + var.s_Kvw[iz, jz, ik] @ var.s_angdisp[jp]
                        var.Msint[ip] += var.s_Kwv[iz, jz, ik] @ var.s_disp[jp] + var.s_Kww[iz, jz, ik] @ var.s_angdisp[jp]

    for ip in range(var.s_nnode[None]):
        if current_pos <= ip < current_pos + BATCH_SIZE and var.s_ibtype[ip] != 200:
            loss[None] += ((var.Fsext[ip][0] - var.Fsint[ip][0])**2 + (var.Msext[ip][0] - var.Msint[ip][0])**2)/var.s_nfree[None]
            loss[None] += ((var.Fsext[ip][1] - var.Fsint[ip][1])**2 + (var.Msext[ip][1] - var.Msint[ip][1])**2)/var.s_nfree[None]
            loss[None] += ((var.Fsext[ip][2] - var.Fsint[ip][2])**2 + (var.Msext[ip][2] - var.Msint[ip][2])**2)/var.s_nfree[None]

    
@ti.kernel
def _update(t: ti.int32, current_pos: ti.int32):
    for ip in range(current_pos, current_pos + BATCH_SIZE):
        if t < 3:
            var.s_disp[ip][t] = fc3.output[0, t, ip-current_pos, 0]*a[t]
        else:
            var.s_angdisp[ip][t-3] = fc3.output[0, t, ip-current_pos, 0]*a[t]

@ti.kernel
def update_FEM():
    for ip in range(var.s_nnode[None]):
        var.s_xm[ip] = var.s_xm0[ip] + var.s_disp[ip]
        var.s_ddisp[ip] = var.s_disp[ip] - var.s_disp_last[ip]
        var.s_dangdisp[ip] = var.s_angdisp[ip] - var.s_angdisp_last[ip]

def update():
    for current_data_offset in range(0, var.s_nnode[None], BATCH_SIZE):
        fill_input_states(current_data_offset)
        fc1.clear()
        fc2.clear()
        fc3.clear()
        # fc4.clear()
        # fc5.clear()
        for i in range(steps):
            fc1.forward(i, input_states)
            fc2.forward(i, fc1.output)
            fc3.forward(i, fc2.output)
            # fc4.forward(i, fc3.output)
            # fc5.forward(i, fc4.output)
            _update(i, current_data_offset)
    update_FEM()

@ti.kernel
def compute_error():
    var.total_MSE[None] = 0.0
    var.disp_MSE[None] = 0.0
    var.angdisp_MSE[None] = 0.0
    var.total_RMSE[None] = 0.0
    var.disp_RMSE[None] = 0.0
    var.angdisp_RMSE[None] = 0.0
    for ip in range(BATCH_SIZE):
        if var.s_ibtype[ip] != 200:
            for i in ti.static(range(3)):
                var.disp_error[ip][i] = ti.abs(var.disp_FEM[ip][i] - var.s_disp[ip][i])
                var.angdisp_error[ip][i] = ti.abs(var.angdisp_FEM[ip][i] - var.s_angdisp[ip][i])
            var.disp_MSE[None] += var.disp_error[ip].dot(var.disp_error[ip])
            var.angdisp_MSE[None] += var.angdisp_error[ip].dot(var.angdisp_error[ip])
    var.disp_MSE[None] /= (3 * var.s_nfree[None])
    var.angdisp_MSE[None] /= (3 * var.s_nfree[None])
    var.disp_RMSE[None] = var.disp_MSE[None]/var.disp_FEM_mean[None]
    var.angdisp_RMSE[None] = var.angdisp_MSE[None]/var.angdisp_FEM_mean[None]
    var.total_MSE[None] = (var.disp_MSE[None] + var.angdisp_MSE[None])/2
    var.total_RMSE[None] = (var.disp_RMSE[None] + var.angdisp_RMSE[None])/2

def closure(current_data_offset):
    for i in range(steps):
        fc1.forward(i, input_states)
        fc2.forward(i, fc1.output)
        fc3.forward(i, fc2.output)
        # fc4.forward(i, fc3.output)
        # fc5.forward(i, fc4.output)
        update_PINN(i, current_data_offset)
    compute_loss(current_data_offset)


def main():
    excel_file = "plate_0.05_g.xlsx"
    df = pd.read_excel(excel_file, sheet_name="Sheet1", engine="openpyxl")
    data = df.to_numpy()
    plot = True
    var.istep[None] = 0
    var.nframe[None] = 0
    Sh_input()
    import_data(data)
    var.nframe[None] += 1

    init_nn_model()

    if TRAIN:
        print("Start training!")
        losses = []
        total_MSE = []
        disp_MSE = []
        angdisp_MSE = []
        total_RMSE = []
        disp_RMSE = []
        angdisp_RMSE = []
        opt_iters = 50000
        n_output = 0
        for opt_iter in range(opt_iters):
            loss_epoch = 0.0
            for current_data_offset in range(0, var.s_nnode[None], BATCH_SIZE):
                update()
                var.istep[None] += 1
                if var.istep[None] == 1:
                    var.s_delta_n.fill(0.0)
                    var.s_ddelta_n.fill(0.0)
                    Sh_K_element()
                else:
                    Sh_coordinate_system_stiff()
                    Sh_update_fiber_quater()
                    Sh_external_force()
                fill_input_states(current_data_offset)
                fc1.clear()
                fc2.clear()
                fc3.clear()
                # fc4.clear()
                # fc5.clear()
                optimizer.zero_grad()
                with ti.ad.Tape(loss=loss):
                    loss[None] = 0.0
                    closure(current_data_offset)
                optimizer.step()
                loss_epoch += loss[None]
            compute_error()
            losses.append(loss_epoch)
            total_MSE.append(var.total_MSE[None])
            disp_MSE.append(var.disp_MSE[None])
            angdisp_MSE.append(var.angdisp_MSE[None])
            total_RMSE.append(var.total_RMSE[None])
            disp_RMSE.append(var.disp_RMSE[None])
            angdisp_RMSE.append(var.angdisp_RMSE[None])
            print(f"opt iter {opt_iter} done. loss: {loss_epoch}, disp_RMSE: {var.disp_RMSE[None]}, angdisp_RMSE: {var.angdisp_RMSE[None]}")

            if opt_iter == 19999 + n_output*10000:
                n_output += 1
                os.makedirs(f"saved_models/{opt_iter}", exist_ok=True)
                fc1.dump_weights(name=f"saved_models/{opt_iter}/fc1_{opt_iter:04}.pkl")
                fc2.dump_weights(name=f"saved_models/{opt_iter}/fc2_{opt_iter:04}.pkl")
                fc3.dump_weights(name=f"saved_models/{opt_iter}/fc3_{opt_iter:04}.pkl")
                # fc4.dump_weights(name=f"saved_models/{opt_iter}/fc4_{opt_iter:04}.pkl")
                # fc5.dump_weights(name=f"saved_models/{opt_iter}/fc5_{opt_iter:04}.pkl")

        data = {
            'Training Loss': losses,
            'Total MSE': total_MSE,
            'Disp MSE': disp_MSE,
            'AngDisp MSE': angdisp_MSE,
            'Total RMSE': total_RMSE,
            'Disp RMSE': disp_RMSE,
            'AngDisp RMSE': angdisp_RMSE,
        }
        ds = pd.DataFrame(data)
        ds.to_excel('loss_history/training_loss.xlsx', index=False)

        plt.plot(range(len(losses)), [math.log10(loss) for loss in losses], label="loss every iteration")
        plt.title("Training Loss")
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss_history/training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(range(len(disp_RMSE)), [math.log10(error) for error in disp_RMSE], label="Disp RMSE")
        plt.plot(range(len(angdisp_RMSE)), [math.log10(error) for error in angdisp_RMSE], label="AngDisp RMSE")
        plt.plot(range(len(total_RMSE)), [math.log10(error) for error in total_RMSE], label="Total RMSE")
        plt.title("Training error")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig('loss_history/Training error.png', dpi=300, bbox_inches='tight')
        plt.show()

        if plot:
            Sh_output_paraview_ascii()

    else:
        print("Start... ")
        update()
        compute_error()
        if plot:
            Sh_output_paraview_ascii()


if __name__ == "__main__":
    main()
