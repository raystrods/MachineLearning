# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:56:21 2016

@author: Ray
"""
import math

class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

class multiplyGate:        
    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)
        return self.utop
        
    def backward(self):
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad
        
class addGate:
    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop
        
    def backward(self):
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad

class sigmoidGate:
    def sig(self, x):
        return 1 / (1 + math.exp(-x))
        
    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop
        
    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1-s)) * self.utop.grad
        
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)


mulg0 = multiplyGate()
mulg1 = multiplyGate()
addg0 = addGate()
addg1 = addGate()
sg0 = sigmoidGate()

def forwardNeuron():
    ax = mulg0.forward(a,x)
    by = mulg1.forward(b,y)
    axpby = addg0.forward(ax, by)
    axpbypc = addg1.forward(axpby, c)
    s = sg0.forward(axpbypc)
    return s

s = forwardNeuron()

print(s.value)

s.grad = 1.0
sg0.backward()
addg1.backward()
addg0.backward()
mulg1.backward()
mulg0.backward()

step_size = 0.01
a.value += step_size * a.grad
b.value += step_size * b.grad
c.value += step_size * c.grad
x.value += step_size * x.grad
y.value += step_size * y.grad
 
s = forwardNeuron()

print(s.value)


   