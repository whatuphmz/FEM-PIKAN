import taichi as ti
import numpy as np
import pickle as pkl
from FEM.Sh_cache import var
"""
Sigmoid High-Order ReLU-Kolmogorov Arnold Networks (HRKAN) Implementation

This module implements three variations of High-Order ReLU-KAN layers using Taichi for GPU acceleration:

1. HRKAN_input: 
   - Input layer variant without batch normalization
   - Directly processes raw inputs using high-order piecewise polynomial activations
   - Suitable for first network layer

2. HRKAN:
   - Standard hidden layer with batch normalization
   - Applies sigmoid to normalized inputs before polynomial activation
   - Designed for intermediate processing

3. HRKAN_output:
   - Output layer with conditional computation
   - Skips processing for specific cases (s_ibtype == 200)
   - Includes batch normalization and sigmoid

Key Features:
- Implements high-order (>1) ReLU-KAN formulations
- Supports multiple independent models (e.g., for 6DOF systems)
- Each "step" (n_steps dimension) represents a separate network instance
- Piecewise polynomial activations defined over configurable grid ranges
- Batched processing for efficient computation
- Weight initialization, forward propagation, and serialization utilities

Structure:
- Grid-based basis functions with (G + k) components per input feature
- High-order polynomial terms controlled by 'order' parameter
- Parameters: coef (weights), phase_low/phases_height (grid boundaries)
- Batch normalization applied in hidden/output layers

Typical Usage:
1. Initialize with model dimensions (input/hidden/output)
2. Configure grid parameters (G, k, range, polynomial order)
3. Call weights_init() for parameter initialization
4. For each epoch, call forward() with input data
5. Access results through output field

Note: The n_steps dimension represents independent network instances 
(not sequential time steps), enabling parallel processing of different DOFs.
"""

real = ti.f32
scalar = lambda: ti.field(dtype=real)

@ti.data_oriented
class HRKAN_input:
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
        G=10,
        k=4,
        grid_range=[-1, 1],
        order=8
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.G = G
        self.k = k
        self.grid = grid_range
        self.order = order
        self.r = (2 * self.G/((self.k + 1) * (self.grid[1] - self.grid[0])))**2

        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.coef = scalar()
        self.phase_low = scalar()
        self.phase_height = scalar()

        self.batch_node.dense(ti.axes(1, 2, 3, 4), (self.n_steps, self.n_hidden, self.n_input, self.G + self.k)).place(self.coef)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_low)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_height)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.coef, self.phase_low, self.phase_height]

    @ti.func
    def sigmoid(self, x):
        return 1/(1 + ti.exp(-x))

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input * (self.G + self.k)))
        for model_id, t, i, j, l in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input, self.G + self.k):
            self.coef[model_id, t, i, j, l] = (ti.random() * 2 - 1) * q1
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_input, self.G + self.k):
            self.phase_low[model_id, t, i, j] = (j - self.k) * (self.grid[1] - self.grid[0])/self.G + self.grid[0]
            self.phase_height[model_id, t, i, j] = self.phase_low[model_id, t, i, j] + (self.k + 1) * (self.grid[1] - self.grid[0])/self.G

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j, l in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input, self.G + self.k):
            self.output[model_id, t, k, i] += self.coef[model_id, t, i, j, l]*(ti.max(0, nn_input[model_id, t, k, j]-self.phase_low[model_id, t, j, l])*ti.max(0, self.phase_height[model_id, t, j, l]-nn_input[model_id, t, k, j])*self.r)**self.order

    @ti.kernel
    def clear(self):
        self.output.fill(0.0)

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
class HRKAN:
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
        G=10,
        k=4,
        grid_range=[0, 1],
        order=8
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.G = G
        self.k = k
        self.grid = grid_range
        self.order = order
        self.r = (2 * self.G/((self.k + 1) * (self.grid[1] - self.grid[0])))**2

        self.mean = ti.field(ti.f32, shape=(self.n_models, self.n_steps, self.n_input), needs_grad=False)
        self.var = ti.field(ti.f32, shape=(self.n_models, self.n_steps, self.n_input), needs_grad=False)
        
        self.input = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_input)).place(self.input)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.coef = scalar()
        self.phase_low = scalar()
        self.phase_height = scalar()

        self.batch_node.dense(ti.axes(1, 2, 3, 4), (self.n_steps, self.n_hidden, self.n_input, self.G + self.k)).place(self.coef)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_low)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_height)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.coef, self.phase_low, self.phase_height]

    @ti.func
    def sigmoid(self, x):
        return 1/(1 + ti.exp(-x))

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input * (self.G + self.k)))
        for model_id, t, i, j, l in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input, self.G + self.k):
            self.coef[model_id, t, i, j, l] = (ti.random() * 2 - 1) * q1
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_input, self.G + self.k):
            self.phase_low[model_id, t, i, j] = (j - self.k) * (self.grid[1] - self.grid[0])/self.G + self.grid[0]
            self.phase_height[model_id, t, i, j] = self.phase_low[model_id, t, i, j] + (self.k + 1) * (self.grid[1] - self.grid[0])/self.G

    @ti.kernel
    def batch_norm(self, t: ti.i32, nn_input: ti.template()):
        for model_id, j in ti.ndrange(self.n_models, self.n_input):
            for k in range(self.batch_size):
                self.mean[model_id, t, j] += nn_input[model_id, t, k, j] / self.batch_size
            for k in range(self.batch_size):
                self.var[model_id, t, j] += (nn_input[model_id, t, k, j] - self.mean[model_id, t, j]) ** 2 / self.batch_size
            for k in range(self.batch_size):
                self.input[model_id, t, k, j] = (nn_input[model_id, t, k, j] - self.mean[model_id, t, j]) / (ti.sqrt(self.var[model_id, t, j]))

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j, l in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input, self.G + self.k):
            self.output[model_id, t, k, i] += self.coef[model_id, t, i, j, l]*(ti.max(0, self.sigmoid(self.input[model_id, t, k, j])-self.phase_low[model_id, t, j, l])*ti.max(0, self.phase_height[model_id, t, j, l]-self.sigmoid(self.input[model_id, t, k, j]))*self.r)**self.order

    @ti.kernel
    def clear(self):
        self.mean.fill(0.0)
        self.var.fill(0.0)
        self.input.fill(0.0)
        self.output.fill(0.0)

    def forward(self, t, nn_input):
        self.batch_norm(t, nn_input)
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
class HRKAN_output:
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
        G=10,
        k=4,
        grid_range=[0, 1],
        order=8
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.G = G
        self.k = k
        self.grid = grid_range
        self.order = order
        self.r = 2 * self.G/((self.k + 1) * (self.grid[1] - self.grid[0]))

        self.mean = ti.field(ti.f32, shape=(self.n_models, self.n_steps, self.n_input), needs_grad=False)
        self.var = ti.field(ti.f32, shape=(self.n_models, self.n_steps, self.n_input), needs_grad=False)
        
        self.input = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_input)).place(self.input)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.coef = scalar()
        self.phase_low = scalar()
        self.phase_height = scalar()

        self.batch_node.dense(ti.axes(1, 2, 3, 4), (self.n_steps, self.n_hidden, self.n_input, self.G + self.k)).place(self.coef)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_low)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.n_input, self.G + self.k)).place(self.phase_height)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.coef, self.phase_low, self.phase_height]

    @ti.func
    def sigmoid(self, x):
        return 1/(1 + ti.exp(-x))

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / (self.n_input * (self.G + self.k)))
        for model_id, t, i, j, l in ti.ndrange(self.n_models, self.n_steps, self.n_hidden, self.n_input, self.G + self.k):
            self.coef[model_id, t, i, j, l] = (ti.random() * 2 - 1) * q1
        for model_id, t, i, j in ti.ndrange(self.n_models, self.n_steps, self.n_input, self.G + self.k):
            self.phase_low[model_id, t, i, j] = (j - self.k) * (self.grid[1] - self.grid[0])/self.G + self.grid[0]
            self.phase_height[model_id, t, i, j] = self.phase_low[model_id, t, i, j] + (self.k + 1) * (self.grid[1] - self.grid[0])/self.G

    @ti.kernel
    def batch_norm(self, t: ti.i32, nn_input: ti.template()):
        for model_id, j in ti.ndrange(self.n_models, self.n_input):
            for k in range(self.batch_size):
                self.mean[model_id, t, j] += nn_input[model_id, t, k, j] / self.batch_size
            for k in range(self.batch_size):
                self.var[model_id, t, j] += (nn_input[model_id, t, k, j] - self.mean[model_id, t, j]) ** 2 / self.batch_size
            for k in range(self.batch_size):
                self.input[model_id, t, k, j] = (nn_input[model_id, t, k, j] - self.mean[model_id, t, j]) / (ti.sqrt(self.var[model_id, t, j]))

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j, l in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input, self.G + self.k):
            if var.s_ibtype[k] == 200:
                self.output[model_id, t, k, i] = 0.0
            else:
                self.output[model_id, t, k, i] += self.coef[model_id, t, i, j, l]*(ti.max(0, self.sigmoid(self.input[model_id, t, k, j])-self.phase_low[model_id, t, j, l])*ti.max(0, self.phase_height[model_id, t, j, l]-self.sigmoid(self.input[model_id, t, k, j]))*self.r)**self.order

    @ti.kernel
    def clear(self):
        self.mean.fill(0.0)
        self.var.fill(0.0)
        self.input.fill(0.0)
        self.output.fill(0.0)

    def forward(self, t, nn_input):
        self.batch_norm(t, nn_input)
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
