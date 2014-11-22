#ifndef NEURON_TARGET_HPP_
#define NEURON_TARGET_HPP_

#include "../../Utility/node_output_itf.hpp"

namespace wwd
{

class CNeuronTarget
    : public INodeOutput<1>
{
public:

    CNeuronTarget(void)
        : INodeOutput()
    {
    }

    ~CNeuronTarget(void)
    {
    }

    inline void backward(void)
    {
        m_output_val = m_input_val - m_input_node[0]->get_output_value();
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        connect_input_node(neuron);
        neuron.connect_output_node(*this);
    }

    inline void set_target( IN double target )
    {
        m_input_val = target;
    }
};

} // namespace wwd

#endif // NEURON_TARGET_HPP_

