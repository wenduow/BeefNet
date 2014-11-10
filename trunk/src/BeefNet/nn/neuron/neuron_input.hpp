#ifndef NEURON_INPUT_HPP_
#define NEURON_INPUT_HPP_

#include "../../utility/node.hpp"

namespace wwd
{

template < uint32 OutputNum >
class CNeuronInput
    : public CNode< 0, OutputNum >
{
public:

    CNeuronInput(void)
        : CNode()
    {
    }

    ~CNeuronInput(void)
    {
    }

    inline void forward(void)
    {
        IForward::m_output = IForward::m_input;
    }

    inline void set_input( IN double input )
    {
        IForward::m_input = input;
    }
};

} // namespace wwd

#endif // NEURON_INPUT_HPP_

