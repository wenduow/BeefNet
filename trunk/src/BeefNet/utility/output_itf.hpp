#ifndef OUTPUT_ITF_HPP_
#define OUTPUT_ITF_HPP_

#include "bp_itf.hpp"

namespace wwd
{

class IFP;

template < uint32 InputNum >
class IOutput
    : public IBP
{
public:

    template < class Input >
    void connect_input( INOUT Input &input, IN bool reverse = true )
    {
        for ( auto &input_iter : m_input )
        {
            if ( NULL == input_iter )
            {
                input_iter = &input;
                break;
            }
        }

        if (reverse)
        {
            input.connect_output( *this, false );
        }
    }

protected:

    IOutput( IN double val = 0.0 )
        : IBP(val)
    {
        for ( auto &input_iter : m_input )
        {
            input_iter = NULL;
        }
    }

    ~IOutput(void)
    {
    }

protected:

    IFP *m_input[InputNum];
};

} // namespace wwd

#endif // OUTPUT_ITF_HPP_

