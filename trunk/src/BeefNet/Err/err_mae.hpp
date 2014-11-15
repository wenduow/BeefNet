#ifndef ERR_MAE_HPP_
#define ERR_MAE_HPP_

#include "err_itf.hpp"

namespace wwd
{

class FErrMAE
    : IErr
{
public:

    FErrMAE(void)
        : IErr()
    {
    }

    ~FErrMAE(void)
    {
    }

    inline double operator()( IN const double *x,
                              IN const double *y,
                              IN uint32 num ) const
    {
        double err = 0.0;

        for ( uint32 i = 0; i < num; ++i )
        {
            err += fabs( x[i] - y[i] );
        }

        return err / (double)num;
    }
};

} // namespace wwd

#endif // ERR_MAE_HPP_

