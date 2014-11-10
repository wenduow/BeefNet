#ifndef PARAM_LM_HPP_
#define PARAM_LM_HPP_

#include "../../utility/type.hpp"

namespace wwd
{

template < uint32 Lambda = 10, uint32 Beta = 10 >
class CParamLM
{
public:

    enum
    {
        lambda = Lambda,
        beta = Beta
    };

private:

    CParamLM(void);
    ~CParamLM(void);
};

}

#endif // PARAM_LM_HPP_

