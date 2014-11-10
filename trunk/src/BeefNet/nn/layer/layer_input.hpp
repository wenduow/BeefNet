#ifndef LAYER_INPUT_HPP_
#define LAYER_INPUT_HPP_

#include "../neuron/neuron_input.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum >
class CLayerInput
{
public:

    enum
    {
        neuron_num = InputNum,
        output_num = OutputNum,
    };

public:

    CLayerInput(void)
    {
    }

    ~CLayerInput(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_neuron )
        {
            i.forward();
        }
    }

    template < class Output >
    void connect_output( INOUT Output &output )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                for ( uint32 k = 0; k < InputNum; ++k )
                {
                    if ( output.get_weight( i, k )
                            .connect_input( m_neuron[j] ) )
                    {
                        m_neuron[j]
                            .connect_output( output.get_weight( i, k ) );
                        break;
                    }
                }
            }
        }
    }

    void set_input( IN const double *input )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_neuron[i].set_input( input[i] );
        }
    }

    inline CNeuronInput<OutputNum> &get_neuron( IN uint32 idx )
    {
        return m_neuron[idx];
    }

private:

    CNeuronInput<OutputNum> m_neuron[InputNum];
};

} // namespace wwd

#endif // LAYER_INPUT_HPP_

