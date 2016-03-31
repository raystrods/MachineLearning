# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:48:57 2016

@author: Ray
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:56:21 2016

@author: Ray
"""
import math
import random

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

class Circuit:
    def __init__(self):
        self.mulg0 = multiplyGate()
        self.mulg1 = multiplyGate()
        self.addg0 = addGate()
        self.addg1 = addGate()

    def forward(self, x, y, a, b,c):
        self.ax = self.mulg0.forward(a,x)
        self.by = self.mulg1.forward(b,y)
        self.axpby = self.addg0.forward(self.ax, self.by)
        self.axpbypc = self.addg1.forward(self.axpby, c)
        return self.axpbypc

    def backward(self, gradient_top):
        self.axpbypc.grad = gradient_top
        self.addg1.backward()
        self.addg0.backward()
        self.mulg1.backward()
        self.mulg0.backward()

class SVM:
    def __init__(self):
        self.a = Unit(1.0, 0.0)
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(-1.0, 0.0)
        self.circuit = Circuit()

    def forward(self, x, y):
        self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
        return self.unit_out
        
    def backward(self, label):
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0
        
        pull = 0.0
        if label == 1 and self.unit_out.value < 1:
            pull = 1.0
        if label == -1 and self.unit_out.value > -1:
            pull = -1.0
            
        self.circuit.backward(pull)
        
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value
        
    def learnFrom(self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.parameterUpdate()
        
    def parameterUpdate(self):
        step_size = 0.05
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad
        
data = []; labels = []
data.append([1.2, 0.7]); labels.append(1)
data.append([-0.3, -0.5]); labels.append(-1)
data.append([3.0, 0.1]); labels.append(1)
data.append([-0.1, -1.0]); labels.append(-1)
data.append([-1.0, 1.1]); labels.append(-1)
data.append([2.1, -3]); labels.append(1)


svm = SVM()

def evalTrainingAccuracy():
    num_correct = 0.0
    for i in range(0,len(data),1):
        xe = Unit(data[i][0], 0.0)
        ye = Unit(data[i][1], 0.0)
        true_label = labels[i]
        
        predicted_label = 1 if svm.forward(xe,ye).value > 0 else -1
        # print 'predicted label :',predicted_label
        # print 'true label :',true_label        
        if predicted_label == true_label:
            num_correct += 1
        # print 'num correct:', num_correct
    return num_correct / len(data)

for iter in range(0,400,1):
    i = int(math.floor(random.random() * len(data)))
    xi = Unit(data[i][0], 0.0)
    yi = Unit(data[i][1], 0.0)
    label = labels[i]
    svm.learnFrom(xi, yi, label)
    
    if iter % 25 == 0:
        # print svm.a.value
        # print svm.b.value
        print 'training accuracy at ',iter,' : ',evalTrainingAccuracy()
        
        
# print svm.a.value
# print svm.b.value
# print svm.c.value
        
        
        
        
        

   