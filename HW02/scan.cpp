#include "scan.h"

//scan function
/*  
    output[0] = arr[0]
    output[1] = arr[0] + arr[1]
    output[2] = arr[0] + arr[1] + arr[2]
    ...
    output[i] = sum( arr[0] + arr[1] + ... + arr[i] )
*/ 
void scan(const float *arr, float *output, std::size_t n){
   // if(n==0) return; //edge case
    output[0] = arr[0];
    for(std::size_t i=1;i<n;i++){
        output[i] = output[i-1] + arr[i];

    }
}
