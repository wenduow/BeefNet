#ifndef ERR_RMSE_HPP_
#define ERR_RMSE_HPP_

#include "err_itf.hpp"

namespace wwd
{

class FErrRMSE
    : IErr
{
public:

    FErrRMSE(void)
        : IErr()
    {
    }

    ~FErrRMSE(void)
    {
    }

    inline double operator()( IN const double *x,
                              IN const double *y,
                              IN uint32 num ) const
    {
        double err = 0.0;

        for ( uint32 i = 0; i < num; ++i )
        {
            err += pow( x[i] - y[i], 2 );
        }

        return sqrt( err / (double)num );
    }
};

} // namespace wwd

#endif // ERR_RMSE_HPP_

