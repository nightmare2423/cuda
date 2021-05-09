#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 64
#define BLOCK_STEP 32
#define BLOCK_NUM 64

#define MOD(a,b) ((a) - (a) / (b) * (b))

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__global__ void bodyForce(Body *p, float dt, int n) {
        //每个线程算一个点的一部分
                int i=MOD(threadIdx.x+blockIdx.x*blockDim.x,n);
                                int nn=n/(BLOCK_STEP*BLOCK_SIZE);
                float Fx = 0.0f;
                                float Fy = 0.0f;
                                float Fz = 0.0f;
                                __shared__ float x_shared[BLOCK_SIZE];
                                __shared__ float y_shared[BLOCK_SIZE];
                                __shared__ float z_shared[BLOCK_SIZE];
                                float xi=p[i].x;
                                float yi=p[i].y;
                                float zi=p[i].z;//储存对应位置的数据
                                float dx,dy,dz,distSqr,invDist,invDist3;
                                int j=(blockIdx.x+blockIdx.x/BLOCK_NUM);

                #pragma unroll
                                while(nn--){
                                        //从当前块开始向前推进
                                        j=MOD(j,BLOCK_NUM);
                                    //获得当前块中的当前相对位置的数据，并写到shared中
                                        x_shared[threadIdx.x]=p[j*BLOCK_SIZE+threadIdx.x].x;
                                        y_shared[threadIdx.x]=p[j*BLOCK_SIZE+threadIdx.x].y;
                                        z_shared[threadIdx.x]=p[j*BLOCK_SIZE+threadIdx.x].z;
                                        //同步锁
                                        __syncthreads();
                                        for(int k=0;k<BLOCK_SIZE;k++){
                        dx = x_shared[k] - xi;
                        dy = y_shared[k] - yi;
                        dz = z_shared[k] - zi;
                        distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                        invDist = rsqrtf(distSqr);
                        invDist3 = invDist * invDist * invDist;
                        Fx += dx * invDist3;
                                                Fy += dy * invDist3;
                                                Fz += dz * invDist3;
                                        }
                                        j+=BLOCK_STEP;
                                        //算完之后才能写入新数据
                                        __syncthreads();
                }
                                //原子加保证数据正确性
                                atomicAdd(&p[i].vx, dt * Fx);
                                atomicAdd(&p[i].vy, dt * Fy);
                                atomicAdd(&p[i].vz, dt * Fz);

}

__global__ void integrate_position(Body *p,float dt,int n){
        int i=threadIdx.x+blockIdx.x*blockDim.x;
                // integrate position
                p[i].x += p[i].vx*dt;
                p[i].y += p[i].vy*dt;
                p[i].z += p[i].vz*dt;
}

int main(const int argc, const char** argv) {

  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2<<11;
  int salt = 0;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  /*
   * This salt is for assessment reasons. Tampering with it will result in automatic failure.
   */

  if (argc > 2) salt = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;
  cudaMallocHost((void **)&buf,bytes);

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host function.
   */
  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
  float *d_buf;
  cudaMalloc((void **)&d_buf,bytes);
  Body *d_p=(Body *)d_buf;

  cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
  /*******************************************************************/

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * as well as the work to integrate the positions.
   */

    bodyForce<<<BLOCK_NUM*BLOCK_STEP,BLOCK_SIZE>>>(d_p, dt, nBodies); // compute interbody forces

  /*
   * This position integration cannot occur until this round of `bodyForce` has completed.
   * Also, the next round of `bodyForce` cannot begin until the integration is complete.
   */
   integrate_position<<<nBodies/BLOCK_SIZE,BLOCK_SIZE>>>(d_p,dt,nBodies);

   if(iter==nIters-1)
       cudaMemcpy(buf,d_buf,bytes,cudaMemcpyDeviceToHost);

  /*******************************************************************/
  // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }



  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
  checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
  checkAccuracy(buf, nBodies);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  salt += 1;
#endif
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */
  cudaFree(buf);
}
