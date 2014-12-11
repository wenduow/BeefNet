#ifndef ERR_MSE_HPP_
#define ERR_MSE_HPP_

#include "err_itf.hpp"

namespace wwd
{

class FErrMSE
    : IErr
{
public:

    FErrMSE(void)
        : IErr()
    {
    }

    ~FErrMSE(void)
    {
    }

    inline double operator()( IN const double *x,
                              IN const double *y,
                              IN uint32 num ) const
    {
        double err = 0.0;

        for ( uint32 i = 0; i < num; ++i )
        {
            err += std::pow( ( x[i] - y[i] ), 2 );
        }

        return err / (double)num;
    }
};

} // namespace wwd

#endif // ERR_MSE_HPP_

