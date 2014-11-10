#ifndef NEURON_HPP_
#define NEURON_HPP_

#include "../../utility/node.hpp"

namespace wwd
{

template < class Xfer, uint32 InputNum, uint32 OutputNum >
class CNeuron
    : public CNode< InputNum, OutputNum >
{
public:

    CNeuron(void)
        : CNode()
    {
    }

    ~CNeuron(void)
    {
    }

    void forward(void)
    {
        IForward::m_input = 0.0;

        for ( const auto &i : m_input_node )
        {
            IForward::m_input += i->get_output();
        }

        IForward::m_output = m_xfer( IForward::m_input );
        IBackward::m_input = m_xfer.derivative( IForward::m_input );
    }

    void backward(void)
    {
        IBackward::m_output = 0.0;

        for ( const auto &i : m_output_node )
        {
            IBackward::m_output += i->get_output();
        }
    }

private:

    CNeuron( IN const CNeuron &other );
    inline CNeuron &operator=( IN const CNeuron &other );

private:

    const Xfer m_xfer;
};

} // namespace wwd

#endif // NEURON_HPP_

