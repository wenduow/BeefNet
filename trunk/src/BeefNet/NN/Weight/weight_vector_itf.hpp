#ifndef WEIGHT_VECTOR_ITF_HPP_
#define WEIGHT_VECTOR_ITF_HPP_

#include "weight.hpp"

namespace wwd
{

template < uint32 InputNum >
class IWeightVector
{
public:

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            if ( i.connect_input_node(neuron) )
            {
                neuron.connect_output_node(i);
                break;
            }
        }
    }

    template < class Neuron >
    void connect_output_neuron( INOUT Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            i.connect_output_node(neuron);
            neuron.connect_input_node(i);
        }

        m_output_neuron = &neuron;
    }

#ifdef _DEBUG
    void print_weight(void)
    {
        for ( auto &i : m_weight )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

protected:

    IWeightVector(void)
        : m_output_neuron(NULL)
    {
    }

    ~IWeightVector(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_weight )
        {
            i.forward();
        }
    }

    void backward(void)
    {
        for ( auto &i : m_weight )
        {
            i.backward();
        }
    }

protected:

    const IBackward *m_output_neuron;
    CWeight m_weight[InputNum];
};

} // namespace wwd

#endif // WEIGHT_VECTOR_ITF_HPP_

