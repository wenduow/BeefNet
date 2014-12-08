#ifndef NODE_INPUT_ITF_HPP_
#define NODE_INPUT_ITF_HPP_

#include "path_forward_itf.hpp"
#include "path_backward_itf.hpp"

namespace wwd
{

template < uint32 OutputNum >
class INodeInput
    : public IPathForward
{
public:

    template < class Output >
    bool connect_output_node( IN const Output &output )
    {
        for ( auto &i : m_output_node )
        {
            if ( !i )
            {
                i = &output;
                return true;
            }
        }

        return false;
    }

protected:

    INodeInput(void)
        : IPathForward()
    {
        for ( auto &i : m_output_node )
        {
            i = NULL;
        }
    }

    ~INodeInput(void)
    {
    }

protected:

    const IPathBackward *m_output_node[OutputNum];
};

} // namespace wwd

#endif // NODE_INPUT_ITF_HPP_

