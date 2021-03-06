#ifndef LAYER_INPUT_HPP_
#define LAYER_INPUT_HPP_

#include "../Neuron/neuron.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum >
class CLayerInput
{
public:

    enum
    {
        hidden_num = InputNum
    };
	
private:

    typedef CNeuronInput<OutputNum> Input;

public:

    CLayerInput(void)
    {
    }

    ~CLayerInput(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_input )
        {
            i.forward();
        }
    }

    void set_input( IN const double *input )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_input[i].set_input( input[i] );
        }
    }

    inline CNeuronInput<OutputNum> &get_hidden_node( IN uint32 idx )
    {
        return m_input[idx];
    }

private:

    Input m_input[InputNum];
};

} // namespace wwd

#endif // LAYER_INPUT_HPP_

