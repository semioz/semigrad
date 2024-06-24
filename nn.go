package semigrad

import "math/rand"

type Neuron struct {
	weights []*Value
	bias    *Value
	nonLin  bool
}

type Layer struct {
	neurons []*Neuron
}

type MLP struct {
	layers []Layer
}

// Creates nin(number of inputs) number of Value objects with random initial weights and a bias(for inactivity)
func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := 0; i < nin; i++ {
		w[i] = NewValue(rand.Float64()*2-1, nil, "")
	}
	return &Neuron{
		weights: w,
		bias:    NewValue(rand.Float64()*2-1, nil, ""),
		nonLin:  true,
	}
}

func (n *Neuron) NeuronParams() []*Value {
	params := make([]*Value, len(n.weights)+1)
	for i, w := range n.weights {
		params[i] = w
	}
	params[len(n.weights)] = n.bias
	return params
}

func (n *Neuron) Forward(inputs []*Value) *Value {
	// Sum of the product of the weights and the inputs
	act := n.bias
	//weighted sum of inputs plus the bias
	for i, w := range n.weights {
		act = act.Add(w.Mul(inputs[i]))
	}
	if n.nonLin {
		//introduces non-linearity into the network, allowing it to learn more complex patterns
		return act.Relu()
	}
	return act
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewNeuron(nin)
	}
	return &Layer{neurons: neurons}
}

func (l *Layer) LayerParams() []*Value {
	params := make([]*Value, 0)
	for _, n := range l.neurons {
		params = append(params, n.NeuronParams()...)
	}
	return params
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)
	layers := make([]Layer, len(sz)-1)
	for i := 0; i < len(sz)-1; i++ {
		layers[i] = *NewLayer(sz[i], sz[i+1])
	}
	return &MLP{layers: layers}
}

func (m *MLP) MLPParams() []*Value {
	params := make([]*Value, 0)
	for _, l := range m.layers {
		params = append(params, l.LayerParams()...)
	}
	return params
}

// TODO
func (m *MLP) Loss() {}

// gotta .zero_grad() before backward
func (m *MLP) ZeroGrad() {
	for _, p := range m.MLPParams() {
		p.grad = 0
	}
}

// Forward pass -> Reset Grad -> Backward pass -> Update(Gradient Descent)
// Minimizing the loss through gradient descent
func (m *MLP) Optimize() {
	// Forward pass: input data is passed through the network. Each layer computes its output using its current weights and biases.
	// Compute loss: The output of the network is compared to the target output using a loss function. A common loss function for regression problems is the mean squared error.

	// Reset Grad

	// Backward pass

	// Update
}
