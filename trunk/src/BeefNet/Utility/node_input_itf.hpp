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
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            if ( !m_output_node[i] )
            {
                m_output_node[i] = &output;
                return true;
            }
        }

        return false;
    }

protected:

    INodeInput(void)
        : IPathForward()
    {
        m_output_node = new const IPathBackward *[OutputNum];

        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_output_node[i] = NULL;
        }
    }

    ~INodeInput(void)
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_output_node[i] = NULL;
        }
        delete[] m_output_node;
        m_output_node = NULL;
    }

protected:

    const IPathBackward **m_output_node;
};

} // namespace wwd

#endif // NODE_INPUT_ITF_HPP_

