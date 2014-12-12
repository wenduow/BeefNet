#ifndef NEURON_HPP_
#define NEURON_HPP_

#include "../../Utility/node_input_itf.hpp"
#include "../../Utility/node_output_itf.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum, class Xfer >
class CNeuron
    : public INodeInput<OutputNum>
    , public INodeOutput<InputNum>
{
private:

    typedef INodeInput<OutputNum> BaseTypeInput;
    typedef INodeOutput<InputNum> BaseTypeOutput;
public:

    CNeuron(void)
        : BaseTypeInput()
        , BaseTypeOutput()
    {
    }

    ~CNeuron(void)
    {
    }

    void forward(void)
    {
        IPathForward::m_input_val = 0.0;

        for ( auto &i : BaseTypeOutput::m_input_node )
        {
            IPathForward::m_input_val += i->get_output_value();
        }

        IPathForward::m_output_val = m_xfer( IPathForward::m_input_val );
    }

    void backward(void)
    {
        IPathBackward::m_input_val = 0.0;

        for ( auto &i : BaseTypeInput::m_output_node )
        {
            IPathBackward::m_input_val += i->get_output_value();
        }

        // calculate delta = f'(net) * sum(delta) from next layer
        IPathBackward::m_output_val
            = m_xfer.derivative( IPathForward::m_input_val )
                               * IPathBackward::m_input_val;
    }

    template < class WeightVector >
    void connect_input_weight_vector( INOUT WeightVector &weight_vector )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            connect_input_node( weight_vector.get_weight(i) );
            weight_vector.get_weight(i).connect_output_node(*this);
        }
    }

private:

    CNeuron( IN const CNeuron &other );
    CNeuron &operator=( IN const CNeuron &other );

private:

    const Xfer m_xfer;
};

} // namespace wwd

#endif // NEURON_HPP_

