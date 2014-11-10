#ifndef NODE_HPP_
#define NODE_HPP_

#include "input_itf.hpp"
#include "output_itf.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum >
class CNode
    : public IInput
    , public IOutput
{
public:

    CNode(void)
        : IInput()
        , IOutput()
    {
        for ( auto &i : m_input_node )
        {
            i = NULL;
        }

        for ( auto &i : m_output_node )
        {
            i = NULL;
        }
    }

    ~CNode(void)
    {
    }

    template < class Node >
    void connect_input( INOUT Node &node, IN bool reverse = true )
    {
        for ( auto &i : m_input_node )
        {
            if ( !i )
            {
                i = &node;
                break;
            }
        }

        if (reverse)
        {
            node.connect_output( *this, false );
        }
    }

    template < class Node >
    void connect_output( INOUT Node &node, IN bool reverse = true )
    {
        for ( auto &i : m_output_node )
        {
            if ( !i )
            {
                i = &node;
                break;
            }
        }

        if (reverse)
        {
            node.connect_input( *this, false );
        }
    }

protected:

    const IInput *m_input_node[InputNum];
    const IOutput *m_output_node[OutputNum];
};

template < uint32 OutputNum >
class CNode< 0, OutputNum >
    : public IInput
{
public:

    CNode(void)
        : IInput()
    {
        for ( auto &i : m_output_node )
        {
            i = NULL;
        }
    }

    ~CNode(void)
    {
    }

    template < class Node >
    void connect_output( INOUT Node &node, IN bool reverse = true )
    {
        for ( auto &i : m_output_node )
        {
            if ( !i )
            {
                i = &node;
                break;
            }
        }

        if (reverse)
        {
            node.connect_input( *this, false );
        }
    }

protected:

    const IOutput *m_output_node[OutputNum];
};

template < uint32 InputNum >
class CNode< InputNum, 0 >
    : public IOutput
{
public:

    CNode(void)
        : IOutput()
    {
        for ( auto &i : m_input_node )
        {
            i = NULL;
        }
    }

    ~CNode(void)
    {
    }

    template < class Node >
    void connect_input( INOUT Node &node, IN bool reverse = true )
    {
        for ( auto &i : m_input_node )
        {
            if ( !i )
            {
                i = &node;
                break;
            }
        }

        if (reverse)
        {
            node.connect_output( *this, false );
        }
    }

protected:

    const IInput *m_input_node[InputNum];
};

} // namespace wwd

#endif // NODE_HPP_

