#ifndef INPUT_ITF_HPP_
#define INPUT_ITF_HPP_

#include "fp_itf.hpp"

namespace wwd
{

class IBP;

template < uint32 OutputNum >
class IInput
    : public IFP
{
public:

    template < class Output >
    void connect_output( INOUT Output &output, IN bool reverse = true )
    {
        for ( auto &output_iter : m_output )
        {
            if ( NULL == output_iter )
            {
                output_iter = &output;
                break;
            }
        }

        if (reverse)
        {
            output.connect_input( *this, false );
        }
    }

protected:

    IInput(void)
        : IFP()
    {
        for ( auto &output_iter : m_output )
        {
            output_iter = NULL;
        }
    }

    ~IInput(void)
    {
    }

protected:

    IBP *m_output[OutputNum];
};

} // namespace wwd

#endif // INPUT_ITF_HPP_

