#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "../neuron/neuron.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 NeuronNum, class Xfer,
           uint32 OutputNum >
class CLayer
{
public:

    enum
    {
        input_num = InputNum,
        neuron_num = NeuronNum
    };

public:

    CLayer(void)
    {
    }

    ~CLayer(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_neuron )
        {
            i.forward();
        }
    }

    void backward(void)
    {
        for ( auto &i : m_neuron )
        {
            i.backward();
        }
    }

    /** connect input weight layer */
    template < class Input >
    void connect_input( INOUT Input &input )
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < Input::input_num; ++j )
            {
                m_neuron[i].connect_input
                    ( input.get_weight_vector(i).get_weight(j) );
            }

            input.get_weight_vector(i).set_output( m_neuron[i] );
        }
    }

    inline CNeuron< Xfer, InputNum, OutputNum > &get_neuron( IN uint32 idx )
    {
        return m_neuron[idx];
    }

private:

    CNeuron< Xfer, InputNum, OutputNum > m_neuron[NeuronNum];
};

} // namespace wwd

#endif // LAYER_HPP_

