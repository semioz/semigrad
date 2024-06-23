package semigrad

import "math"

// Tanh activation function to be used in the hidden layers
// derivative of tanh(x) is 1 - tanh²(x)
func (v *Value) Tanh() *Value {
	t := math.Tanh(v.data)
	out := NewValue(t, []*Value{v}, "tanh")
	out._backward = func() {
		v.grad += (1 - t*t) * out.grad
	}
	return out
}

// rectified linear unit. activation function f(x) = max(0, x), returns 0 for negative inputs and the input itself for positive values.
func (v *Value) Relu() *Value {
	out := NewValue(math.Max(0, v.data), []*Value{v}, "relu")
	out._backward = func() {
		if v.data > 0 {
			v.grad += out.grad
		}
	}
	return out
}

// f(x) = 1 / (1 + e^(-x)) activation function
// outputs values between 0 and 1, often used in the output layer for binary classification
func (v *Value) Sigmoid() *Value {
	sig := 1 / (1 + math.Exp(-(v.data)))
	out := NewValue(sig, []*Value{v}, "sigmoid")
	out._backward = func() {
		v.grad += sig * (1 - sig) * out.grad
	}
	return out
}
