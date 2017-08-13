# NVIDIA-NCCL-EXAMPLE
Here is the simple and short way to teaching how to code NCCL and complier

You need to have more than 1 gpu graph card

In my case , I have four GTX 1080

# complier with
After install nccl with the https://github.com/NVIDIA/nccl

~~~
#just add the link library
nvcc nccl.cu -lnccl
~~~

nccl.cu is what you want to complier with nccl code

# Usage
## _Number of gpu_
#nessary

Need to get current amounts number of GPU
~~~
#Get current amounts number of GPU

int nGPUs = 0;
cudaGetDeviceCount(&nGPUs);
printf("nGPUs = %d\n",nGPUs);
~~~

## _list the gpu_
#nessary

DeviceList is a array that put the number of the GPUnum
~~~
#List GPU Device


int *DeviceList;  
DeviceList = (int *)malloc( nGPUs * sizeof(int));
for (int i = 0; i < nGPUs; ++i){
  DeviceList[i] = i;
}
~~~

## _Init_
#nessary

NCCL Init will lunch the GPU

Can be seen as we poweron the not in opertation GPU card

According to your number of GPU

In my case , It cost about 0.8s~1.0s
~~~
#NCCL Init

ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nGPUs);  
cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nGPUs);
ncclCommInitAll(comms, nGPUs, DeviceList);
~~~
## _GPU status_
#choose

~~~
#Get GPU status
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
~~~
