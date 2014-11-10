#ifndef NEURON_TARGET_HPP_
#define NEURON_TARGET_HPP_

#include "../../utility/node.hpp"

namespace wwd
{

class CNeuronTarget
    : public CNode< 1, 0 >
{
public:

    CNeuronTarget(void)
        : CNode()
    {
    }

    ~CNeuronTarget(void)
    {
    }

    inline void backward(void)
    {
        IBackward::m_output = IBackward::m_input
                            - m_input_node[0]->get_output();
    }

    inline void set_target( IN double target )
    {
        IBackward::m_input = target;
    }
};

} // namespace wwd

#endif // NEURON_TARGET_HPP_

