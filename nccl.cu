#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <nccl.h>
__global__ void show(int *in,int i){
    in[threadIdx.x] = i;
    printf("%d\n",in[threadIdx.x]);
}

__global__ void show_al(int *in){
    //printf("%d\n",in[threadIdx.x]);
}

int main(int argc, char* argv[]) {

  /*Get current amounts number of GPU*/
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);
  printf("nGPUs = %d\n",nGPUs);

  /*List GPU Device*/
  int *DeviceList;  
  DeviceList = (int *)malloc( nGPUs * sizeof(int));
  for (int i = 0; i < nGPUs; ++i){
      DeviceList[i] = i;
  }
  
  /*NCCL Init*/
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nGPUs);  
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);
  /*Get GPU status*/
  printf("# Using devices\n");
  for (int g = 0; g < nGPUs; g++) {
      int cudaDev;
      int rank;
      cudaDeviceProp prop;
      ncclCommCuDevice(comms[g], &cudaDev);
      ncclCommUserRank(comms[g], &rank);

      cudaGetDeviceProperties(&prop, cudaDev);
      printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev, prop.pciBusID, prop.name);
  }
  printf("\n");

  /*Malloc the data*/
  int data_size = 2000000000 ;
  int* data;
  data = (int*)malloc(data_size * sizeof(int));
  int **d_data;
  d_data = (int**)malloc(nGPUs * sizeof(int*));
  int **d_data_al;
  d_data_al = (int**)malloc(nGPUs * sizeof(int*));


  for(int i = 0; i < data_size; i++){
      data[i] = i;
  }


  for(int g = 0; g < nGPUs; g++) {
      char busid[32] = {0};
      cudaDeviceGetPCIBusId(busid, 32, DeviceList[g]);
      printf("# Rank %d using device %d [%s]\n", g, DeviceList[g], busid);

      cudaSetDevice(DeviceList[g]);
      cudaStreamCreate(&s[g]);
      cudaMalloc(&d_data[g], data_size * sizeof(int));
      cudaMalloc(&d_data_al[g], data_size * sizeof(int));

      if(g == 0) {
          cudaMemcpy(d_data[g], data, data_size * sizeof(int),cudaMemcpyHostToDevice);
      }
  }

  

  // GPU Bcast
  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      ncclBcast(d_data[i], data_size, ncclInt, 0, comms[i], s[i]);
  }


  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      printf("This is device %d \n",i);
      //show<<<1,8>>>(d_data[i],i);
      cudaThreadSynchronize();
  }
  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      cudaStreamSynchronize(s[i]);
  }

  printf("Bcast Done!\n");

  // GPU Allgather
  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      ncclAllGather(d_data[i], 1, ncclInt, d_data_al[i], comms[i], s[i]);
  }
  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      printf("This is device %d \n",i);
      //show_al<<<1,8>>>(d_data_al[i]);
      cudaThreadSynchronize();
  }
  for (int i = 0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      cudaStreamSynchronize(s[i]);
  }

  printf("Allgather Done!\n");


  for(int i=0; i < nGPUs; ++i) {
      cudaSetDevice(DeviceList[i]);
      cudaStreamDestroy(s[i]);
  }

  printf("StreamDestroy Done!\n");

  free(s);
  cudaFree(d_data);
  cudaFree(d_data_al);
}

