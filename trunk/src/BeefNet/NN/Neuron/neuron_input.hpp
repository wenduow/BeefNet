#ifndef NEURON_INPUT_HPP_
#define NEURON_INPUT_HPP_

#include "../../Utility/node_input_itf.hpp"

namespace wwd
{

template < uint32 OutputNum >
class CNeuronInput
    : public INodeInput<OutputNum>
{
private:

    typedef INodeInput<OutputNum> BaseTypeInput;

public:

    CNeuronInput(void)
        : BaseTypeInput()
    {
    }

    ~CNeuronInput(void)
    {
    }

    inline void forward(void)
    {
        BaseTypeInput::m_output_val = BaseTypeInput::m_input_val;
    }

    inline void set_input( IN double input )
    {
        BaseTypeInput::m_input_val = input;
    }
};

} // namespace wwd

#endif // NEURON_INPUT_HPP_

