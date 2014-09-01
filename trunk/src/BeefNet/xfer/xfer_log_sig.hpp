#ifndef XFER_LOG_SIG_HPP_
#define XFER_LOG_SIG_HPP_

#include <cmath>
#include "xfer_itf.hpp"

namespace wwd
{

class FXferLogSig : public IXfer
{
public:

    FXferLogSig(void)
        : IXfer()
    {
    }

    ~FXferLogSig(void)
    {
    }

    inline double operator()( IN double val ) const
    {
        return ( 1.0 / ( 1.0 + exp(-val) ) );
    }

    inline double derivative( IN double val ) const
    {
        return ( operator()(val) * ( 1 - operator()(val) ) );
    }
};

}

#endif // XFER_LOG_SIG_HPP_

