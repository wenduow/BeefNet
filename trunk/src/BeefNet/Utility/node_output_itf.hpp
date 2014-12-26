#ifndef NODE_OUTPUT_ITF_HPP_
#define NODE_OUTPUT_ITF_HPP_

#include "path_forward_itf.hpp"
#include "path_backward_itf.hpp"

namespace wwd
{

template < uint32 InputNum >
class INodeOutput
    : public IPathBackward
{
public:

    template < class Input >
    bool connect_input_node( IN const Input &input )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            if ( !m_input_node[i] )
            {
                m_input_node[i] = &input;
                return true;
            }
        }

        return false;
    }

protected:

    INodeOutput(void)
        : IPathBackward()
    {
        m_input_node = new const IPathForward *[InputNum];

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_input_node[i] = NULL;
        }
    }

    ~INodeOutput(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_input_node[i] = NULL;
        }
        delete[] m_input_node;
        m_input_node = NULL;
    }

protected:

    const IPathForward **m_input_node;
};

} // namespace wwd

#endif // NODE_OUTPUT_ITF_HPP_

