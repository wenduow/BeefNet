#ifndef PARAM_BP_HPP_
#define PARAM_BP_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 LearnRate = 500 >
class EParamBP
{
public:

    enum
    {
        learn_rate = LearnRate
    };

private:

    EParamBP(void);
    ~EParamBP(void);
};

} // namespace wwd

#endif // PARAM_BP_HPP_

