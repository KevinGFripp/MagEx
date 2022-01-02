#ifndef REDUCE_TEMPLATES_CUH
#define REDUCE_TEMPLATES_CUH

template <class T>
void reduce(int size, int threads, int blocks,
    int whichKernel, T* d_idata, T* d_odata);

template <class T>
void
reduce_Energy(int size, int threads, int blocks,
    int whichKernel, T* M_idata, T* H_idata, T* E_odata);

#endif // !REDUCE_TEMPLATES_CUH
