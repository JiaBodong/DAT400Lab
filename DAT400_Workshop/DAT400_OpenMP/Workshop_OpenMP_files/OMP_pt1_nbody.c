#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>


#define ERR 1e-3
#define DIFF(x,y) ((x-y)<0? y-x : x-y)
#define FPNEQ(x,y) (DIFF(x,y)>ERR ? 1 : 0)
int test(int N, float * sol, float * p, float * ax, float * ay) {
  int i;
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i],p[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+N],ax[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+2 * N], ay[i]))
      return 0;
  }
  return 1;
}

int main(int argc, char** argv) {
  // Initialize
  int pow = (argc > 1)? atoi(argv[1]) : 14;
  int N = 1 << pow;
  int i, j;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  float* x =  (float*)malloc(N * sizeof(float));
  float* y =  (float*)malloc(N * sizeof(float));
  float* m =  (float*)malloc(N * sizeof(float));
  float* p =  (float*)malloc(N * sizeof(float));
  float* ax = (float*)malloc(N * sizeof(float));
  float* ay = (float*)malloc(N * sizeof(float));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48() / N;
    p[i] = ax[i] = ay[i] =  0;
  }

  printf("Running for problem size N: %d\n", N);

  //Timers
  double ts, tf;

  //Serial version 
  printf("Running serial......................................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    for (j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }
  tf = omp_get_wtime();
  printf("Time: %.4lfs\n", tf - ts);

  //Copying solution for correctness check
  float* sol = (float*)malloc(3 * N * sizeof(float));
  memcpy(sol, p, N * sizeof(float));
  memcpy(sol + N, ax, N * sizeof(float));
  memcpy(sol+ 2 * N, ay, N * sizeof(float));


  //TODO: SPMD - Question 1 - Parallelize the outer loop 

  printf("Running parallel (outer loop).......................\n");
  ts = omp_get_wtime();
#pragma omp parallel
{
  int i;
  int id = omp_get_thread_num();
  int numthreads = omp_get_num_threads();
  int part_size = N / numthreads;
  
  int start = id * part_size;
  int end = ( id + 1 == numthreads)?
  	N : start + part_size;
  

  for (i=start; i<end; i++) {            //FIXME: Parallelize
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    for (j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }
  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");
}
  //TODO: SPMD - Question 2 - Parallelize the inner loop 

  printf("Running parallel (inner loop).......................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) 
  {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    float pii[12];
    float axx[12];
    float ayy[12];
    
    #pragma omp parallel num_threads(12)
    {	
  	  int numthreads = omp_get_num_threads();
  	  int part_size = N / numthreads;
  	  int id = omp_get_thread_num();
  	  int start = id * part_size;
  	  int end = ( id + 1 == numthreads)? N : start + part_size;
	   //printf("numthreads:%d \n",&numthreads);
      for (int j=start; j<end; j++) {       //FIXME: Parallelize
    	//printf("j:%d\n",&j);
      	 float dx = x[j] - xi;
      	 float dy = y[j] - yi;
      	 float R2 = dx * dx + dy * dy + EPS2;
      	 float invR = 1.0f / sqrtf(R2);
      	 float invR3 = m[j] * invR * invR * invR;
      	//printf("j:%d \n",&j);
   	     pii[id] += m[j] * invR;
   	     axx[id] += dx * invR3;
   	     ayy[id] += dy * invR3;
   	//printf("id:%d,pii[id]:%f \n",&j,&pii[j]);
   	    }
    }
     
    for(int i=0; i< 12; i++)
    {
      pi += pii[i];
      axi += axx[i];
      ayi += ayy[i];
    }
       
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  //TODO: SPMD - Question 3 - Parallelize the inner loop and avoid false sharing

  printf("Running parallel (inner loop without false sharing).\n");
  ts = omp_get_wtime();
 
   for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    float pii[N*16];
    float axx[N*16];
    float ayy[N*16];
    
     #pragma omp parallel 
     {	
  	int j;
  	int id = omp_get_thread_num();
  	int numthreads = omp_get_num_threads();
  	int part_size = N / numthreads;
  
  	int start = id * part_size;
  	int end = ( id + 1 == numthreads)?
  		N : start + part_size;
  		
    	for (j=start; j<end; j++) {       //FIXME: Parallelize
      	float dx = x[j] - xi;
      	float dy = y[j] - yi;
      	float R2 = dx * dx + dy * dy + EPS2;
      	float invR = 1.0f / sqrtf(R2);
      	float invR3 = m[j] * invR * invR * invR;
   	pii[id * 16]=m[j] * invR;
   	axx[id * 16]=dx * invR3;
   	ayy[id * 16]=dy * invR3;
   	}
     }
     
     for(int i=0; i< N; i++)
     {
       pi += pii[i*16];
       axi += axx[i*16];
       ayi += ayy[i*16];
     }
       
     p[i] = pi;
     ax[i] = axi;
     ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  
  free(x);
  free(y);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(sol);
  return 0;
}

