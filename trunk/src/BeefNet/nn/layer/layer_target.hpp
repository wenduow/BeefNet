#ifndef LAYER_TARGET_HPP_
#define LAYER_TARGET_HPP_

#include "../neuron/neuron_target.hpp"

namespace wwd
{

template < uint32 OutputNum >
class CLayerTarget
{
public:

    enum
    {
        neuron_num = OutputNum,
    };


public:

    CLayerTarget(void)
    {
    }

    ~CLayerTarget(void)
    {
    }

    void backward(void)
    {
        for ( auto &i : m_neuron )
        {
            i.backward();
        }
    }

    template < class Input >
    void connect_input( INOUT Input &input )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_neuron[i].connect_input( input.get_neuron(i) );
        }
    }

    inline CNeuronInput<OutputNum> &get_neuron( IN uint32 idx )
    {
        return m_neuron[idx];
    }

    void set_target( IN const double *target )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_neuron[i].set_target( target[i] );
        }
    }

private:

    CNeuronTarget m_neuron[OutputNum];
};

} // namespace wwd

#endif // LAYER_TARGET_HPP_

