#ifndef WEIGHT_LM_HPP_
#define WEIGHT_LM_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < class Param >
class CWeightLM
    : public IWeight
{
public:

    CWeightLM(void)
        : IWeight()
    {
    }

    ~CWeightLM(void)
    {
    }
};

} // namespace wwd

#endif // WEIGHT_LM_HPP_

