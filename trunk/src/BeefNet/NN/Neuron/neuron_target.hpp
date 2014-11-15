#ifndef NEURON_TARGET_HPP_
#define NEURON_TARGET_HPP_

#include "../../Utility/output_itf.hpp"

namespace wwd
{

class CNeuronTarget
    : public IOutput<1>
{
public:

    CNeuronTarget(void)
        : IOutput()
    {
    }

    ~CNeuronTarget(void)
    {
    }

    inline void backward(void)
    {
        m_output_val = m_input_val - m_input_node[0]->get_output_val();
    }

    template < class Input >
    inline void connect_input_neuron( INOUT Input &input )
    {
        connect_input_node(input);
        input.connect_output_node(*this);
    }

    inline void set_target( IN double target )
    {
        m_input_val = target;
    }
};

} // namespace wwd

#endif // TARGET_HPP_

