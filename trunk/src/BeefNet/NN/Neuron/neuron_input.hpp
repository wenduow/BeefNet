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

    typedef INodeInput<OutputNum> BaseType;
public:

    CNeuronInput(void)
        : BaseType()
    {
    }

    ~CNeuronInput(void)
    {
    }

    inline void forward(void)
    {
        BaseType::m_output_val = BaseType::m_input_val;
    }

    inline void set_input( IN double input )
    {
        BaseType::m_input_val = input;
    }
};

} // namespace wwd

#endif // NEURON_INPUT_HPP_

