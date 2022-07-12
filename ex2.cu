#include "ex2.h"
#include <new>
#include <cuda/atomic>


__device__ void prefix_sum(int arr[256], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    // trick: allow running multiple times (modulu) & skip threads (negative arr_size)
    // arr_size must be the same for all or __syncthreads will cause deadlock
    tid      = (arr_size > 0)? tid % arr_size : 0; 
    arr_size = (arr_size > 0)? arr_size       : -arr_size;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return; 
}

/**
* map between a thread to its tile, total tile number is TILES_COUNT^2
*/
__device__ int get_tile_id(int index) {
    int line = index / IMG_WIDTH;
    int col  = index % IMG_WIDTH;
    line = line / TILE_HEIGHT; // round down
    col  = col / TILE_WIDTH; 
    return line * TILE_COUNT + col;
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


__device__ void process_image(uchar *all_in, uchar *all_out, uchar *maps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tnum = blockDim.x;

    __shared__ int histograms[IMG_TILES][256];

    for (int i = tid; i < IMG_TILES*256; i += tnum) { // set zero
        ((int*) histograms)[i] = 0;
    }
    __syncthreads();

    for (int index = tid; index < IMG_SIZE; index += tnum) { // calc histograms
	int tile = get_tile_id(index);
	uchar pix_val = all_in[index];
	int *hist = &(histograms[tile][pix_val]);
        atomicAdd(hist, 1);
    }
    __syncthreads();

    // run prefix sum in each tile --- ASSUME: tnum  >= 256
    for (int run=0; run < (IMG_TILES/(tnum/256)+1); run++) { // enforce same amount of entries to prefix_sum
        int tile = (tid/256) + run*(tnum/256);
        if (tile >= IMG_TILES) 
            prefix_sum(NULL, -256);  // keep internal syncthread from blocking the rest
        else 
            prefix_sum(&(histograms[tile][0]), 256);
    }

//    for (int i = 0; i < IMG_TILES ; i++) {
//	    prefix_sum(histograms[bid][i], 256);
//	    __syncthreads();
//    }

    __syncthreads();

    // create map
    for (int i = tid; i < IMG_TILES*256; i += tnum) { 
        int cdf = ((int*) histograms)[i];
//        maps[MAP_SIZE*bid + i] = (uchar) ((((double)cdf)*255)/(TILE_WIDTH*TILE_HEIGHT)); // cast will round down
	    uchar map_value =(((double)cdf) / (TILE_WIDTH*TILE_HEIGHT)) * 255;
	    maps[i] = map_value;
    }
    __syncthreads();

    interpolate_device(maps, all_in, all_out);

    __syncthreads();
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

// ----------------------------------

struct task_context {
    uchar *d_image_in;
    uchar *d_image_out;
    uchar *d_maps; 
};

class streams_server : public image_processing_server
{
private:
    cudaStream_t streams[STREAM_COUNT];
    task_context contexts[STREAM_COUNT];
    int ids[STREAM_COUNT];
public:
    streams_server()
    {
        for(int i=0; i<STREAM_COUNT; i++) {
            cudaStreamCreate(&streams[i]);

            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_image_in),  IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_image_out), IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_maps), MAP_SIZE));

            ids[i] = NO_ID;
        }
    }

    ~streams_server() override
    {
        for(int i=0; i<STREAM_COUNT; i++) {
            cudaFree(contexts[i].d_image_in);
            cudaFree(contexts[i].d_image_out);
            cudaFree(contexts[i].d_maps);

            cudaStreamDestroy(streams[i]);
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        // not safe for MT in cpu
        for( int s=0; s<STREAM_COUNT; s++) {
            // check empty
            if(ids[s] != NO_ID)
                continue;
            ids[s] = img_id;
	    //CUDA_CHECK(cudaStreamQuery(streams[s])); // debug only
            // start
            CUDA_CHECK(cudaMemcpyAsync(contexts[s].d_image_in, img_in, IMG_SIZE, cudaMemcpyHostToDevice, streams[s]));
            process_image_kernel<<<1,THREAD_NUM,0,streams[s]>>>(contexts[s].d_image_in, contexts[s].d_image_out, contexts[s].d_maps); 
            CUDA_CHECK(cudaMemcpyAsync(img_out, contexts[s].d_image_out, IMG_SIZE, cudaMemcpyDeviceToHost, streams[s]));
	    //printf("queued\n");
            return true;
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {

        // TODO query (don't block) streams for any completed requests.
        for( int s=0; s<STREAM_COUNT; s++) {
            if(ids[s] == NO_ID)
                continue;
            cudaError_t status = cudaStreamQuery(streams[s]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                *img_id = ids[s];
                ids[s] = NO_ID; // mark for reuse
		//printf("dequeued %d @ %d\n", *img_id, s);
                return true;
            case cudaErrorNotReady:
                break; // and continue loop
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// implement a lock
// only in gpu, different for in and out

typedef cuda::atomic<int, cuda::thread_scope_device> gpu_atomic_int;
__global__ void init_lock(gpu_atomic_int* _lock) { new(_lock) gpu_atomic_int(0); }
__global__ void destroy_lock(gpu_atomic_int* _lock) {_lock->~gpu_atomic_int(); }

__device__ void lock(gpu_atomic_int *l) {
    while(l->exchange(1, cuda::memory_order_relaxed));
    cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
}
__device__ void unlock(gpu_atomic_int *l) {
    l->store(0, cuda::memory_order_release);
}

// implement a MPMC queue - not single one
    template<class T> // TODO continue extracting methods
    ring_buffer<T>::ring_buffer(size_t n) : _head(0), _tail(0) {
        N = n; // must be a power of 2
        cudaMallocHost(&_mailbox, N*sizeof(T));
        cudaMallocHost(&terminate, sizeof(int));
        *terminate = 0;

        CUDA_CHECK(cudaMalloc(&_lock, sizeof(gpu_atomic_int)));
        init_lock<<<1,1>>>(_lock);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<class T> 
    ring_buffer<T>::~ring_buffer() {
        cudaFreeHost(_mailbox);
	    cudaFreeHost(terminate);

        destroy_lock<<<1,1>>>(_lock);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(_lock);
    }

    template <class T>
    __device__ __host__ int ring_buffer<T>::push(const T &data, int abort) {
        // abort on fail option for cpu side
        int tail = _tail.load(cuda::memory_order_relaxed);
        while (tail - _head.load(cuda::memory_order_acquire) == N  && !*terminate && !abort);
        if(*terminate ||  tail - _head.load(cuda::memory_order_acquire) == N) return 0;
        _mailbox[_tail % N] = data;
        _tail.store(tail + 1, cuda::memory_order_release);
        return 1;
    }
    template <class T>
    __device__ __host__ int ring_buffer<T>::pop( T* result, int abort) {
        int head = _head.load(cuda::memory_order_relaxed);
        while (_tail.load(cuda::memory_order_acquire) == _head  && !*terminate && !abort);
        if(*terminate ||  _tail.load(cuda::memory_order_acquire) == _head) return 0;
        *result = _mailbox[_head % N];
        _head.store(head + 1, cuda::memory_order_release);
        return 1;
    }

    template <class T>
    __device__ T ring_buffer<T>::gpu_pop() {
        lock(_lock);
        T res;
        pop(&res);
        unlock(_lock);
        return res;
    }
    template <class T>
    __device__ void ring_buffer<T>::gpu_push(const T& data) {
        lock(_lock);
        push(data);
        unlock(_lock);
    }
// end of ring_buffer definition

struct task_info {
    uchar *d_image_in;
    uchar *d_image_out;
    int id;
    //uchar *d_maps; 
};


// TODO implement the persistent kernel each tb is a different calculator (remember the use of barriers)
__global__ void run_cores(ring_buffer<task_info> *buffer_in, ring_buffer<int> *buffer_out, uchar *d_maps) {
    __shared__ task_info task;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    while(true) {
        if(tid == 0) task = buffer_in->gpu_pop(); 
        __syncthreads(); // only 1 thread in block need to access lock
        if(*(buffer_in->terminate)) {
            if(tid == 0) // reduce number of pcie writes
                buffer_out->terminate[0] = 1; 
            return; 
        } 
        process_image(task.d_image_in, task.d_image_out, d_maps + MAP_SIZE*bid);
        if(tid == 0) buffer_out->gpu_push(task.id);
        // no need to sync. will wait in the next cycle
    }
}

// TODO implement a function for calculating the threadblocks count
int calc_tb() {
    return 128; // TODO implement
}

// -------
class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    ring_buffer<task_info> *buffer_in;
    ring_buffer<int>       *buffer_out; // pinned
    uchar *maps; 

public:
    queue_server(int threads)
    {
        int tb_num = calc_tb();
        char *temp;
        // initialize host state
        // maps:
        cudaMallocHost(&maps, MAP_SIZE*tb_num*sizeof(uchar));
        // in:
        cudaMallocHost(&temp, sizeof(ring_buffer<task_info>));
        buffer_in = new(temp) ring_buffer<task_info>(64*tb_num);
        // out:
        cudaMallocHost(&temp, sizeof(ring_buffer<int>));
        buffer_out = new(temp) ring_buffer<int>(64*tb_num);

        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        run_cores<<<tb_num, threads>>>(buffer_in, buffer_out, maps);
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        buffer_in->terminate[0] = 1;
        buffer_out->terminate[0] = 1; // just in case
        CUDA_CHECK(cudaDeviceSynchronize());
        buffer_in->~ring_buffer();
        buffer_out->~ring_buffer();
        cudaFreeHost(buffer_in);
        cudaFreeHost(buffer_out);
        cudaFreeHost(maps);
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        task_info task;
        task.d_image_in = img_in;
        task.d_image_out = img_out;
        task.id = img_id;
        return buffer_in->push(task, 1);
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        int status = buffer_out->pop(img_id, 1);
	// if (status) printf("next\n");
	return status;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
