import taichi as ti
import numpy as np
import pickle as pkl
from FEM.Sh_cache import var
"""
Multilayer Perceptron (MLP) Network Layers

This module implements various MLP layer types optimized for GPU computation using Taichi. 
The layers are designed for neural networks with multiple independent models (e.g., for 6DOF systems) 
where 'n_steps' represents independent network instances rather than temporal steps.

Implemented Layer Types:
1. Linear_input:
   - Input layer without activation (or with optional GELU)
   - Standard matrix multiplication + bias
   - Xavier weight initialization

2. Linear:
   - Hidden layer with optional GELU activation
   - Standard matrix multiplication + bias
   - Supports activation toggle

3. Linear_output:
   - Output layer with conditional processing
   - Special handling for ibtype=200 cases
   - Tanh activation for outputs (except specific indices)
   - Weight initialization adapted for output layer

Common Features:
- Batch processing support for efficient computation
- Multiple independent network instances (n_models)
- GPU acceleration through Taichi's data-oriented design
- Weight saving/loading (pickle format)
- Parameter management utilities
- Clear functionality to reset internal states
- Support for both activated and linear outputs

Activation Details:
- Uses Gaussian Error Linear Unit (GELU) when activation=True
- Implements GELU approximation: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3))
- Output layer uses tanh activation for constrained outputs

Usage:
1. Initialize layer with dimensions (input/hidden/output) and activation flag
2. Call weights_init() for parameter initialization
3. For each epoch, call forward() with input data
4. Access results through output field
5. Use dump/load_weights for model serialization

Note: All implementations leverage Taichi's parallel computation capabilities for GPU acceleration.
"""

dtype_f_np = np.float32
real = ti.f32
scalar = lambda: ti.field(dtype=real)


@ti.data_oriented
class Linear_input:
    def __init__(
        self,
        n_models,
        batch_size,
        n_steps,
        n_input,
        n_hidden,
        n_output,
        needs_grad=False,
        activation=False,
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.steps_node = self.batch_node.dense(ti.j, self.n_steps)
        self.n_hidden_node = self.steps_node.dense(ti.k, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.l, self.n_input)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input))
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input):
            self.weights1[model_id, t, i, j] = (ti.random() * 2 - 1) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
            self.hidden[model_id, t, k, i] += self.weights1[model_id, t, i, j] * nn_input[model_id, t, k, j]
        if ti.static(self.activation):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = 0.5*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i])*(1+ti.tanh(ti.sqrt(2/3.1415926)*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i] + 0.044715*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i])**3)))
        else:
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i]

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.0
        for I in ti.grouped(self.output):
            self.output[I] = 0.0

    def forward(self, t, nn_input):
        self._forward(t, nn_input)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w = w.to_numpy()
            w_val.append(w[0])
        with open(name, "wb") as f:
            pkl.dump(w_val, f)

    def load_weights(self, name="save.pkl", model_id=0):
        with open(name, "rb") as f:
            w_val = pkl.load(f)
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @staticmethod
    @ti.kernel
    def copy_from_numpy(dst: ti.template(), src: ti.types.ndarray(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]

@ti.data_oriented
class Linear:
    def __init__(
        self,
        n_models,
        batch_size,
        n_steps,
        n_input,
        n_hidden,
        n_output,
        needs_grad=False,
        activation=False,
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.steps_node = self.batch_node.dense(ti.j, self.n_steps)
        self.n_hidden_node = self.steps_node.dense(ti.k, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.l, self.n_input)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input))
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input):
            self.weights1[model_id, t, i, j] = (ti.random() * 2 - 1) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
            self.hidden[model_id, t, k, i] += self.weights1[model_id, t, i, j] * nn_input[model_id, t, k, j]
        if ti.static(self.activation):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = 0.5*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i])*(1+ti.tanh(ti.sqrt(2/3.1415926)*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i] + 0.044715*(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i])**3)))
        else:
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i]

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.0
        for I in ti.grouped(self.output):
            self.output[I] = 0.0

    def forward(self, t, nn_input):
        self._forward(t, nn_input)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w = w.to_numpy()
            w_val.append(w[0])
        with open(name, "wb") as f:
            pkl.dump(w_val, f)

    def load_weights(self, name="save.pkl", model_id=0):
        with open(name, "rb") as f:
            w_val = pkl.load(f)
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @staticmethod
    @ti.kernel
    def copy_from_numpy(dst: ti.template(), src: ti.types.ndarray(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]

@ti.data_oriented
class Linear_output:
    def __init__(
        self,
        n_models,
        batch_size,
        n_steps,
        n_input,
        n_hidden,
        n_output,
        needs_grad=False,
        activation=False,
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.steps_node = self.batch_node.dense(ti.j, self.n_steps)
        self.n_hidden_node = self.steps_node.dense(ti.k, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.l, self.n_input)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input))
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input):
            self.weights1[model_id, t, i, j] = (ti.random() * 2 - 1) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
            self.hidden[model_id, t, k, i] += self.weights1[model_id, t, i, j] * nn_input[model_id, t, k, j]
        if ti.static(self.activation):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                if var.s_ibtype[k] == 200:
                    self.output[model_id, t, k, i] = 0.0
                else:
                    self.output[model_id, t, k, i] = ti.tanh(self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i])
        else:
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                if var.s_ibtype[k] == 200:
                    self.output[model_id, t, k, i] = 0.0
                else:
                    self.output[model_id, t, k, i] = self.hidden[model_id, t, k, i] + self.bias1[model_id, t, i]

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.0
        for I in ti.grouped(self.output):
            self.output[I] = 0.0

    def forward(self, t, nn_input):
        self._forward(t, nn_input)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w = w.to_numpy()
            w_val.append(w[0])
        with open(name, "wb") as f:
            pkl.dump(w_val, f)

    def load_weights(self, name="save.pkl", model_id=0):
        with open(name, "rb") as f:
            w_val = pkl.load(f)
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @staticmethod
    @ti.kernel
    def copy_from_numpy(dst: ti.template(), src: ti.types.ndarray(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]
