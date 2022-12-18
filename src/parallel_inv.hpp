#ifndef PARALLEL_INV_HPP_
#define PARALLEL_INV_HPP_


#include "user_types.hpp"

i_real_matrix inv_ref_PP(const i_real_matrix &matG, const bool usePermute);

void showMatrix_PP(const i_real_matrix &matG, const char *describe, bool matlabFormat);


#endif /* LIB_TESTING_HPP_ */
