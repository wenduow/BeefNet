#ifndef NODE_OUTPUT_ITF_HPP_
#define NODE_OUTPUT_ITF_HPP_

#include "path_backward_itf.hpp"

class IPathForward;

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
        for ( auto &i : m_input_node )
        {
            if ( !i )
            {
                i = &input;
                return true;
            }
        }

        return false;
    }

protected:

    INodeOutput(void)
        : IPathBackward()
    {
        for ( auto &i : m_input_node )
        {
            i = NULL;
        }
    }

    ~INodeOutput(void)
    {
    }

protected:

    const IPathForward *m_input_node[InputNum];
};

} // namespace wwd

#endif // NODE_OUTPUT_ITF_HPP_

