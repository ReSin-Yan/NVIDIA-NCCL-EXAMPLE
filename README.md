# NVIDIA-NCCL-EXAMPLE
Here is the simple and short way to teaching how to code NCCL and complier

You need to have more than 1 gpu graph card

In my case , I have four GTX 1080

# Complier with
After install nccl with the https://github.com/NVIDIA/nccl

~~~
#just add the link library -lnccl
nvcc nccl.cu -lnccl
~~~

nccl.cu is what you want to complier with nccl code

# Usage
nessary is nccl needed

choose can omitted
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

## _Data malloc_
#nessary

use the cudaSetDevice to change "is working" GPU

And then , at different GPU , malloc the space

Last , cudaMemcpy the loacl data to the device 0

~~~
#Malloc the data
int data_size = 2000000000 ;
int* data;
data = (int*)malloc(data_size * sizeof(int));
int **d_data;
d_data = (int**)malloc(nGPUs * sizeof(int*));

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

  if(g == 0) {
    cudaMemcpy(d_data[g], data, data_size * sizeof(int),cudaMemcpyHostToDevice);
  }
}
~~~

## _NCCL Testing for Bcast_
#nessary

There are many Collectives with nccl

I use the Bcast by example

~~~
#GPU Bcast
for (int i = 0; i < nGPUs; ++i) {
  cudaSetDevice(DeviceList[i]);
  ncclBcast(d_data[i], data_size, ncclInt, 0, comms[i], s[i]);
}

#Test the Bcast whether run
for (int i = 0; i < nGPUs; ++i) {
  cudaSetDevice(DeviceList[i]);
  printf("This is device %d \n",i);
  show<<<1,8>>>(d_data[i],i);
  cudaThreadSynchronize();
}
#At the end of the process
#Must need to wait all the GPU task finish
#Use the cudaSetDevice and cudaStreamSynchronize to make sure all the GPU task finish
for (int i = 0; i < nGPUs; ++i) {
  cudaSetDevice(DeviceList[i]);
  cudaStreamSynchronize(s[i]);
}
~~~
