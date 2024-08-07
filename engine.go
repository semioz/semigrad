package semigrad

import (
	"math"
)

// Stores a single scalar value and its gradient
// Backward function is used update the gradients of the childs - Chain rule
type Value struct {
	data, grad float64
	prev       []*Value
	op         string
	backward   func()
}

func (v Value) GetValue() float64 {
	return v.data
}

func (v Value) GetGrad() float64 {
	return v.grad
}

func (v *Value) ResetGrad() {
	v.grad = 0
}

func NewValue(data float64, children []*Value, op string) *Value {
	return &Value{
		data:     data,
		grad:     0,
		prev:     children,
		op:       op,
		backward: func() {},
	}
}

func (v *Value) Add(other interface{}) *Value {
	var otherVal *Value
	switch o := other.(type) {
	case *Value:
		otherVal = o
	case float64:
		otherVal = NewValue(o, []*Value{}, "")
	}

	out := NewValue(v.data+otherVal.data, []*Value{v, otherVal}, "+")
	out.backward = func() {
		v.grad += out.grad
		otherVal.grad += out.grad
	}
	return out
}

func (v *Value) Sub(other interface{}) *Value {
	var otherVal *Value
	switch o := other.(type) {
	case *Value:
		otherVal = o
	case float64:
		otherVal = NewValue(o, []*Value{}, "")
	}

	out := NewValue(v.data-otherVal.data, []*Value{v, otherVal}, "-")
	out.backward = func() {
		v.grad += out.grad
		otherVal.grad -= out.grad
	}
	return out
}

func (v *Value) Mul(other interface{}) *Value {
	var otherVal *Value
	switch o := other.(type) {
	case *Value:
		otherVal = o
	case float64:
		otherVal = NewValue(o, []*Value{}, "")
	}

	out := NewValue(v.data*otherVal.data, []*Value{v, otherVal}, "*")
	out.backward = func() {
		v.grad += otherVal.data * out.grad
		otherVal.grad += v.data * out.grad
	}
	return out
}

func (v *Value) Exp() *Value {
	out := NewValue(math.Exp(v.data), []*Value{v}, "exp")
	out.backward = func() {
		v.grad += math.Exp(v.data) * out.grad
	}
	return out
}

func (v *Value) Div(other interface{}) *Value {
	var otherVal *Value
	switch o := other.(type) {
	case *Value:
		otherVal = o
	case float64:
		otherVal = NewValue(o, []*Value{}, "")
	}

	out := NewValue(v.data/otherVal.data, []*Value{v, otherVal}, "/")
	out.backward = func() {
		v.grad += out.grad / otherVal.data
		otherVal.grad -= v.data / (otherVal.data * otherVal.data) * out.grad
	}
	return out
}

func (v *Value) Pow(other float64) *Value {
	out := NewValue(math.Pow(v.data, other), []*Value{v}, "pow")
	out.backward = func() {
		v.grad += other * math.Pow(v.data, other-1) * out.grad
	}
	return out
}

func (v *Value) Backward() {
	//store our topological ordering
	var topo []*Value
	visited := make(map[*Value]bool)

	var buildTopo func(*Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.prev {
				buildTopo(child)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(v)

	// Go one variable at a time and apply the chain rule to get its gradient
	v.grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}
