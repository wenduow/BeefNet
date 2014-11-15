#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "../Neuron/neuron_input.hpp"
#include "../Neuron/neuron.hpp"
#include "../Weight/weight.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 NeuronNum,
           uint32 OutputNum,
           class Xfer,
           template < uint32, class > class Weight,
           class Param >
class CLayer
{
public:

    enum
    {
        hidden_num = NeuronNum
    };

public:

    CLayer(void)
    {
        connect_inner();
    }

    ~CLayer(void)
    {
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].forward();
            m_neuron[i].forward();
        }
    }

    void backward(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_neuron[i].backward();
            m_weight_vector[i].backward();
        }
    }

    void update(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.update();
        }
    }

    template < class Layer >
    void connect_input_layer( INOUT Layer &layer )
    {
        for ( uint32 i = 0; i < Layer::hidden_num; ++i )
        {
            for ( uint32 j = 0; j < NeuronNum; ++j )
            {
                m_weight_vector[j]
                    .connect_input_neuron( layer.get_hidden_node(i) );
            }
        }
    }

    inline CNeuron< InputNum, OutputNum, Xfer> &
        get_hidden_node( IN uint32 idx )
    {
        return m_neuron[idx];
    }

#ifdef _DEBUG
    void print_weight(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].connect_output_neuron( m_neuron[i] );
        }
    }

private:

    Weight< InputNum, Param > m_weight_vector[NeuronNum];
    CNeuron< InputNum, OutputNum, Xfer> m_neuron[NeuronNum];
};

} // namespace wwd

#endif // LAYER_HPP_

