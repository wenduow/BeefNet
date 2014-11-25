#ifndef PARAM_QP_HPP_
#define PARAM_QP_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 LearnRate = 50, uint32 FactMax = 1750 >
class EParamQP
{
private:

    EParamQP(void);
    ~EParamQP(void);

public:

    static const double learn_rate;
    static const double fact_max;
};

template < uint32 LearnRate /** = 500 */, uint32 FactMax /** = 1750 */ >
const double EParamQP< LearnRate, FactMax >::learn_rate
    = (double)LearnRate / 1000.0;

template < uint32 LearnRate /** = 500 */, uint32 FactMax /** = 1750 */ >
const double EParamQP< LearnRate, FactMax >::fact_max
    = (double)FactMax / 1000.0;

} // namespace wwd

#endif // PARAM_QP_HPP_

