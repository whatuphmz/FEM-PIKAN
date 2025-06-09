import taichi as ti
import numpy as np


@ti.data_oriented
class AMSGrad():
    def __init__(self,params,lr = 0.001,beta1 = 0.9,beta2 = 0.999):
        self.params = params
        self.lr = lr
        self.lr_t = ti.field(dtype=ti.f32, shape=())
        self.k = 0
        self.v = None
        self.v_hat = None
        self.m = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self):
        self.iter[None] += 1
        if self.v == None:
            self.v = {}
            self.v_hat = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                self.v_hat[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v_hat[k].fill(0.0)
                k += 1
        if self.m == None:
            self.m = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.m[k] = ti.field(dtype=ti.f32, shape=shape)
                self.m[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1
        
    @ti.kernel
    def _step(self, w: ti.template()):
        if self.iter[None] < 20000:
            self.lr_t[None] = self.lr + (1e-4 - self.lr)*(1 - self.iter[None]/20000)**6
        else:
            self.lr_t[None] = self.lr
        for I in ti.grouped(w):
            self.m[self.k][I] = (self.m[self.k][I] * self.beta1) + ((1 - self.beta1) * w.grad[I])
            self.v[self.k][I] = (self.v[self.k][I] * self.beta2) + ((1 - self.beta2) * w.grad[I] ** 2)
            self.v_hat[self.k][I] = ti.max(self.v_hat[self.k][I], self.v[self.k][I])
            w[I] -= self.lr_t[None]  * ((self.m[self.k][I]) / (ti.sqrt(self.v_hat[self.k][I]) + 1e-7))
        
    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class AdaMax():
    def __init__(self,params,lr = 0.001,beta1 = 0.9,beta2 = 0.999):
        self.params = params
        self.lr = lr
        self.k = 0
        self.v = None
        self.m = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self):
        self.iter[None] += 1
        if self.v == None:
            self.v = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                k += 1
        if self.m == None:
            self.m = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.m[k] = ti.field(dtype=ti.f32, shape=shape)
                self.m[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1
        
    @ti.kernel
    def _step(self, w: ti.template()):
        lr_t = self.lr * ti.sqrt(1.0 - self.beta2 ** self.iter[None]) / (1.0 - self.beta1 ** self.iter[None])
        for I in ti.grouped(w):
            self.m[self.k][I] = (self.m[self.k][I] * self.beta1) + ((1 - self.beta1) * w.grad[I])
            self.v[self.k][I] = ti.max(self.v[self.k][I] * self.beta2, ti.abs(w.grad[I]))
            w[I] -= (self.lr  * self.m[self.k][I]/(1.0 - self.beta1 ** self.iter[None])) / (ti.sqrt(self.v[self.k][I]) + 1e-7)
        
    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class AdamW():
    def __init__(self,params,lr = 0.001,beta1 = 0.9,beta2 = 0.999, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.k = 0
        self.v = None
        self.m = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

    def step(self):
        self.iter[None] += 1
        if self.v == None:
            self.v = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                k += 1
        if self.m == None:
            self.m = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.m[k] = ti.field(dtype=ti.f32, shape=shape)
                self.m[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1
        
    @ti.kernel
    def _step(self, w: ti.template()):
        lr_t = self.lr * ti.sqrt(1.0 - self.beta2 ** self.iter[None]) / (1.0 - self.beta1 ** self.iter[None])
        for I in ti.grouped(w):
            self.m[self.k][I] = (self.m[self.k][I] * self.beta1) + ((1 - self.beta1) * w.grad[I])
            self.v[self.k][I] = (self.v[self.k][I] * self.beta2) + ((1 - self.beta2) * w.grad[I] ** 2)
            w[I] -= (self.lr  * self.m[self.k][I]/(1.0 - self.beta1 ** self.iter[None])) / (ti.sqrt(self.v[self.k][I]/(1.0 - self.beta2 ** self.iter[None])) + 1e-7) + self.lr * self.weight_decay * w[I]
        
    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class NAdam():
    def __init__(self,params,lr = 0.001,beta1 = 0.9,beta2 = 0.999):
        self.params = params
        self.lr = lr
        self.k = 0
        self.v = None
        self.m = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self):
        self.iter[None] += 1
        if self.v == None:
            self.v = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                k += 1
        if self.m == None:
            self.m = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.m[k] = ti.field(dtype=ti.f32, shape=shape)
                self.m[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1
        
    @ti.kernel
    def _step(self, w: ti.template()):
        lr_t = self.lr * ti.sqrt(1 - self.beta2 ** self.iter[None]) / (1 - self.beta1 ** self.iter[None])
        for I in ti.grouped(w):
            self.m[self.k][I] = (self.m[self.k][I] * self.beta1) + ((1 - self.beta1) * w.grad[I])
            self.v[self.k][I] = (self.v[self.k][I] * self.beta2) + ((1 - self.beta2) * w.grad[I] ** 2)
            w[I] -= (self.lr  * (self.beta1 * self.m[self.k][I]/(1 - self.beta1 ** self.iter[None]) + (1 - self.beta1) * w.grad[I]/(1 - self.beta1 ** self.iter[None]))) / (ti.sqrt(self.v[self.k][I]/(1 - self.beta2 ** self.iter[None])) + 1e-7)
        
    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class Adam():
    def __init__(self,params,lr = 0.001,beta1 = 0.9,beta2 = 0.999):
        self.params = params
        self.lr = lr
        self.k = 0
        self.v = None
        self.m = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self):
        self.iter[None] += 1
        if self.v == None:
            self.v = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                k += 1
        if self.m == None:
            self.m = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.m[k] = ti.field(dtype=ti.f32, shape=shape)
                self.m[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1
        
    @ti.kernel
    def _step(self, w: ti.template()):
        lr_t = self.lr * ti.sqrt(1.0 - self.beta2 ** self.iter[None]) / (1.0 - self.beta1 ** self.iter[None])
        for I in ti.grouped(w):
            self.m[self.k][I] = (self.m[self.k][I] * self.beta1) + ((1 - self.beta1) * w.grad[I])
            self.v[self.k][I] = (self.v[self.k][I] * self.beta2) + ((1 - self.beta2) * w.grad[I] ** 2)
            w[I] -= (self.lr  * self.m[self.k][I]/(1.0 - self.beta1 ** self.iter[None])) / (ti.sqrt(self.v[self.k][I]/(1.0 - self.beta2 ** self.iter[None])) + 1e-7)
        
    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class RMSProp():
    def __init__(self,params,lr = 0.01,decay_rate = 0.9):
        self.params = params
        self.lr = lr
        self.h = None
        self.decay_rate = decay_rate

    def step(self):
        if self.h == None:
            self.h = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.h[k] = ti.field(dtype=ti.f32, shape=shape)
                self.h[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            self.h[self.k][I] = self.h[self.k][I] * self.decay_rate + (1 - self.decay_rate) * w.grad[I] ** 2
            w[I] -= self.lr  * w.grad[I] / (ti.sqrt(self.h[self.k][I]) + 1e-7)

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)
            
            
@ti.data_oriented
class Adagrad():
    def __init__(self,params,lr = 0.01):
        self.params = params
        self.lr = lr
        self.h = None
        self.k = 0

    def step(self):
        if self.h == None:
            self.h = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.h[k] = ti.field(dtype=ti.f32, shape=shape)
                self.h[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            self.h[self.k][I] += w.grad[I] ** 2
            w[I] -= self.lr  * w.grad[I] / (ti.sqrt(self.h[self.k][I]) + 1e-7)

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)

            
@ti.data_oriented
class Momentum():
    def __init__(self,params,lr = 0.01,momentum = 0.9):
        self.params = params
        self.lr = lr
        self.v = None#先不考虑初始优化问题，vanilla版本
        self.momentum = momentum

    def step(self):
        if self.v == None:
            self.v = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.v[k] = ti.field(dtype=ti.f32, shape=shape)
                self.v[k].fill(0.0)
                k += 1
        self.k = 0
        for w in self.params:
            self._step(w)
            self.k += 1

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            self.v[self.k][I] = self.v[self.k][I] * self.momentum - self.lr * w.grad[I]
            w[I] += self.v[self.k][I]

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class SGD:
    def __init__(self, params, lr, beta1 = 0.9, beta2 = 0.999):
        self.params = params
        self.lr = lr
        self.w_last = None
        self.grad_last = None
        self.iter = ti.field(dtype=ti.i32, shape=())
        self.iter[None] = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, line_search):
        self.iter[None] += 1
        if self.w_last == None:
            self.w_last = {}
            self.grad_last = {}
            k = 0
            for w in self.params:
                shape = w.shape
                self.w_last[k] = ti.field(dtype=ti.f32, shape=shape)
                self.w_last[k].fill(0.0)
                self.grad_last[k] = ti.field(dtype=ti.f32, shape=shape)
                self.grad_last[k].fill(0.0)
                k += 1

        self.k = 0
        for w in self.params:
            self.get_grad(w)
            self.k += 1

        self.grid_line_search(line_search)

    @ti.kernel
    def get_grad(self, w: ti.template()):
        for I in ti.grouped(w):
            self.w_last[self.k][I] = w[I]
            self.grad_last[self.k][I] = w.grad[I]
            # print(w.grad[I])

    def grid_line_search(self, line_search):
        grid = np.linspace(0, 30, 31)
        steps = 0.5**grid
        loss = np.zeros(len(steps))
        for i in range(len(steps)):
            self.k = 0
            for w in self.params:
                self._step_line(w, steps, i)
                self.k += 1
            loss[i] = line_search()
            print(loss[i])
        step_line = steps[np.nanargmin(loss)]

        self.k = 0
        for w in self.params:
            self._step(w, step_line)
            self.k += 1

    @ti.kernel
    def _step_line(self, w: ti.template(), steps: ti.types.ndarray(), n: int):
        for I in ti.grouped(w):
            w[I] = self.w_last[self.k][I] - self.lr * steps[n]  * self.grad_last[self.k][I]

    @ti.kernel
    def _step(self, w: ti.template(), step_line: ti.f32):
        lr_t = self.lr
        # if 200000 < self.iter[None] < 300000:
        #     lr_t = self.lr * 0.1 / (1.0 - self.beta1 ** self.iter[None])
        # elif self.iter[None] > 300000:
        #     lr_t = self.lr * 0.01 / (1.0 - self.beta1 ** self.iter[None])
        for I in ti.grouped(w):
            w[I] = self.w_last[self.k][I] - lr_t * step_line * w.grad[I]

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)

