#ifndef OUTPUT_ITF_HPP_
#define OUTPUT_ITF_HPP_

#include "backward_itf.hpp"

namespace wwd
{

class IForward;

template < uint32 InputNum >
class IOutput
    : public IBackward
{
public:

    template < class Node >
    bool connect_input_node( IN const Node &node )
    {
        for ( auto &i : m_input_node )
        {
            if ( !i )
            {
                i = &node;
                return true;
            }
        }

        return false;
    }

protected:

    IOutput(void)
        : IBackward()
    {
        for ( auto &i : m_input_node )
        {
            i = NULL;
        }
    }

    ~IOutput(void)
    {
    }

protected:

    const IForward* m_input_node[InputNum];
};

} // namespace wwd

#endif // OUTPUT_ITF_HPP_

