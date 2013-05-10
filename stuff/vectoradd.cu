#include <iostream>
#include <sys/time.h>

#define N 16

__global__ void add(int *a, int *b, int *c)
{
    int i = blockIdx.x;
    if(i < N)
        c[i] = a[i] + b[i];
}

void add_host(int *a, int *b, int *c)
{
    for(int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main (void)
{
    // variables to store host and device data
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    // allocate memory on the GPU
    cudaMalloc( (void **) &dev_a, N * sizeof(int) );
    cudaMalloc( (void **) &dev_b, N * sizeof(int) );
    cudaMalloc( (void **) &dev_c, N * sizeof(int) );
    
    // fill the arrays with data
    for(int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy arrays a and b to the device
    cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );

    // print the current time
    struct timeval tv;
    struct timezone tz;
    struct tm *tm;
    gettimeofday(&tv, &tz);
    tm = localtime(&tv.tv_sec);
    printf(" %d:%02d:%02d %d \n", tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec);
    
    // Do the addition operation
    add<<<N,1>>>( dev_a, dev_b, dev_c );
    //add_host((int *)a, (int *)b, (int *)c);

    // print current time
    gettimeofday(&tv, &tz);
    tm = localtime(&tv.tv_sec);
    printf(" %d:%02d:%02d %d \n", tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec);

    // copy the array 'c' back from the device
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );
    
    // display the results
    for(int i = 0; i<N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
   
    // free the memory used on the device
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
 
    return 0;
}
