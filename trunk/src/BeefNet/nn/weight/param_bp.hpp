#ifndef PARAM_BP_HPP_
#define PARAM_BP_HPP_

#include "../../utility/type.hpp"

namespace wwd
{

template < uint32 LearnRate = 500 >
class CParamBP
{
public:

    enum
    {
        learn_rate = LearnRate
    };

private:

    CParamBP(void);
    ~CParamBP(void);
};

}

#endif // PARAM_BP_HPP_

