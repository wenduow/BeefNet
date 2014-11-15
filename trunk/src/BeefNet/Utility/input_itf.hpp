#ifndef INPUT_ITF_HPP_
#define INPUT_ITF_HPP_

#include "forward_itf.hpp"

namespace wwd
{

class IBackward;

template < uint32 OutputNum >
class IInput
    : public IForward
{
public:

    template < class Node >
    bool connect_output_node( IN const Node &node )
    {
        for ( auto &i : m_output_node )
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

    IInput(void)
        : IForward()
    {
        for ( auto &i : m_output_node )
        {
            i = NULL;
        }
    }

    ~IInput(void)
    {
    }

protected:

    const IBackward *m_output_node[OutputNum];
};

} // namespace wwd

#endif // INPUT_ITF_HPP_

