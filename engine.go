package semigrad

// Stores a single scalar value and its gradient
// Backward function is used update the gradients of the childs - Chain rule
type Value struct {
	data, grad float64
	_prev      []*Value
	_op        string
	_backward  func()
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
	v._backward = func() {
		v.grad += out.grad
		other.grad += out.grad
	}
	return out
}

func (v *Value) Sub(other *Value) *Value {
	out := NewValue(v.data-other.data, []*Value{v, other}, "-")
	v._backward = func() {
		v.grad += out.grad
		other.grad -= out.grad
	}
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data*other.data, []*Value{v, other}, "*")
	v._backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

func (v *Value) Backward() {
	// Topological order all of the children in the graph
	// implement later
}
