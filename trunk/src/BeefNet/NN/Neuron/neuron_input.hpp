#ifndef NEURON_INPUT_HPP_
#define NEURON_INPUT_HPP_

#include "../../Utility/input_itf.hpp"

namespace wwd
{

template < uint32 OutputNum >
class CNeuronInput
    : public IInput<OutputNum>
{
public:

    CNeuronInput(void)
        : IInput()
    {
    }

    ~CNeuronInput(void)
    {
    }

    inline void forward(void)
    {
        m_output_val = m_input_val;
    }

    inline void set_input( IN double input )
    {
        m_input_val = input;
    }
};

} // namespace wwd

#endif // NEURON_INPUT_HPP_

