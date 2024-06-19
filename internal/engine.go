package semigrad

import "fmt"

// Value stores a single scalar value and its gradient
type Value struct {
	data, grad float64
	_prev      []*Value
	_op        string
}

func NewValue(data float64, children []*Value, op string) *Value {
	return &Value{
		data:  data,
		grad:  0,
		_prev: children,
		_op:   op,
	}
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.data+other.data, []*Value{v, other}, "+")
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data*other.data, []*Value{v, other}, "*")
	return out
}

// manuel backprop...
// TODO 3810
func ManBackprop() {
	h := 0.0001

	a := NewValue(2.0, nil, "")
	b := NewValue(-3.0, nil, "")
	c := NewValue(10.0, nil, "")
	e := a.Mul(b)
	d := e.Add(c)
	f := NewValue(-2.0, nil, "")
	L := d.Mul(f)
	L.grad = 1.0
	// we know that L = d*f
	// dL/dd = f = -2
	// dL/df = d = 4
	// f.grad = 4
	// d.grad = dL/df * dL/df = -2 * 1 = -2
	L1 := L.data

	a = NewValue(2.0, nil, "")
	b = NewValue(-3.0, nil, "")
	c = NewValue(10.0, nil, "")
	e = a.Mul(b)
	d = e.Add(c)
	f = NewValue(-2.0, nil, "")
	L = d.Mul(f)
	L.grad = 1.0
	L2 := L.data + h

	fmt.Println((L2 - L1) / h)
}
