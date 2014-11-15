#ifndef XFER_TAN_SIG_HPP_
#define XFER_TAN_SIG_HPP_

#include <cmath>
#include "xfer_itf.hpp"

namespace wwd
{

class FXferTanSig : public IXfer
{
public:

    FXferTanSig(void)
        : IXfer()
    {
    }

    ~FXferTanSig(void)
    {
    }

    inline double operator()( IN double val ) const
    {
        return tanh(val);
    }

    inline double derivative( IN double val ) const
    {
        return 1.0 - pow( tanh(val), 2 );
    }
};

}

#endif // XFER_TAN_SIG_HPP_

