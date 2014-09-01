#ifndef XFER_LNR_HPP_
#define XFER_LNR_HPP_

#include "xfer_itf.hpp"

namespace wwd
{

class FXferLnr : public IXfer
{
public:

    FXferLnr(void)
        : IXfer()
    {
    }

    ~FXferLnr(void)
    {
    }

    inline double operator()( IN double val ) const
    {
        return val;
    }

    inline double derivative( IN double val ) const
    {
        (double)val;

        return 1.0;
    }
};

}

#endif // XFER_LNR_HPP_

