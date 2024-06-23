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

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewNeuron(nin)
	}
	return &Layer{neurons: neurons}
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)
	layers := make([]Layer, len(sz)-1)
	for i := 0; i < len(sz)-1; i++ {
		layers[i] = *NewLayer(sz[i], sz[i+1])
	}
	return &MLP{layers: layers}
}
