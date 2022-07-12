///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>


/* Abstract base class for both parts of the exercise */
class image_processing_server
{
public:
    virtual ~image_processing_server() {}

    /* Enqueue an image for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) = 0;

    /* Checks whether an image has completed processing. If so, set img_id
     * accordingly, and return true. */
    virtual bool dequeue(int *img_id) = 0;
};

template <class T> class ring_buffer {
protected:
    size_t N;
    T *_mailbox;
    cuda::atomic<size_t> _head, _tail;
    gpu_atomic_int *_lock; // in device memory
public:
    int *terminate; // release all threads (assigned once & no need to lock)


    ring_buffer(size_t n);
    virtual ~ring_buffer();
    __device__ __host__ int push(const T &data, int abort=0);
    __device__ __host__ int pop( T* result, int abort=0);

    __device__ T gpu_pop();
    __device__ void gpu_push(const T& data);
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////

