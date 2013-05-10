#include "julia.h"

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex( float a, float b ) : r(a), i(b) {}
    __device__ float magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, int dim_x, int dim_y )
{
    const float scale = 1.0;
    float jx = scale * (float)(dim_x/2 - x)/(dim_x/2);
    float jy = scale * (float)(dim_y/2 - y)/(dim_y/2);
    
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    
    int i = 0;
    for(i = 0; i < 200; i++)
    {
        a = a * a + c;
        if(a.magnitude2() > 1000)
            return 0;
    }
    
    return 1;
}

__global__ void julia_kernel( unsigned char *bitmap )
{
    // map from threadIdx/blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    
    // now calculate the value at that position
    int juliaValue = julia( x, y, gridDim.x, gridDim.y );
    bitmap[offset*4 + 0] = 0;
    bitmap[offset*4 + 1] = 255 * juliaValue;
    bitmap[offset*4 + 2] = 0;
    bitmap[offset*4 + 3] = 255;
}

void julia_set( int width, int height, unsigned char *bitmap )
{
    // Allocate the GPU bitmap
    unsigned char *dev_bitmap = NULL;
    cudaMalloc((void**)&dev_bitmap, width * height * 4);
    
    // Run the Julia Set kernel
    dim3 grid(width, height);
    julia_kernel<<<grid,1>>>( dev_bitmap );
    
    // Copy data back to host
    cudaMemcpy(bitmap, dev_bitmap, width * height * 4, cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(dev_bitmap);
}