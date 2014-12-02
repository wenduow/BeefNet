#ifndef PARAM_BP_HPP_
#define PARAM_BP_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 LearnRate = 500 >
class EParamBP
{
private:

    EParamBP(void);
    ~EParamBP(void);

public:

    static const double learn_rate;
};

template < uint32 LearnRate /** = 500 */ >
const double EParamBP<LearnRate>::learn_rate = (double)LearnRate / 1000.0;

} // namespace wwd

#endif // PARAM_BP_HPP_

