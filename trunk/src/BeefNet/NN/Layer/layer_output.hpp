#ifndef LAYER_OUTPUT_HPP_
#define LAYER_OUTPUT_HPP_

#include "../Neuron/neuron_target.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 OutputNum,
           class Xfer,
           template < uint32, class > class Weight,
           class Param >
class CLayerOutput
{
public:

    CLayerOutput(void)
    {
        connect_inner();
    }

    ~CLayerOutput(void)
    {
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_weight_vector[i].forward();
            m_output[i].forward();
        }
    }

    void backward(void)
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_target[i].backward();
            m_output[i].backward();
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
            for ( uint32 j = 0; j < OutputNum; ++j )
            {
                m_weight_vector[j]
                    .connect_input_neuron( layer.get_hidden_node(i) );
            }
        }
    }

    void set_target( IN const double *target )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_target[i].set_target( target[i] );
        }
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
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_weight_vector[i].connect_output_neuron( m_output[i] );
            m_target[i].connect_input_neuron( m_output[i] );
        }
    }

private:

    Weight< InputNum, Param > m_weight_vector[OutputNum];
    CNeuron< InputNum, 1, Xfer> m_output[OutputNum];
    CNeuronTarget m_target[OutputNum];
};

} // namespace wwd

#endif // LAYER_OUTPUT_HPP_

