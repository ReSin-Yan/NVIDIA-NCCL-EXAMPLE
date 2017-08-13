# NVIDIA-NCCL-EXAMPLE
Here is the simple and teaching way to teaching how to code NCCL and complier

You need to have more than 1 gpu graph card

# complier with
After install nccl with the https://github.com/NVIDIA/nccl

~~~
#just add the link library
nvcc nccl.cu -lnccl
~~~

nccl.cu is what you want to complier with nccl code

# Usage
Nedd to get current amounts number of GPU
~~~
#Get current amounts number of GPU

int nGPUs = 0;
cudaGetDeviceCount(&nGPUs);
printf("nGPUs = %d\n",nGPUs);
~~~
