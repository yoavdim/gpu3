/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};



// -- direct rdma part --

struct rings_packet {
    client_remote_ring from_client_to_server; // task_info
    client_remote_ring to_client_from_server; // id
    // TODO add char* for server's img_in&out start of array
    uchar *server_img_in_offset, *server_img_out_offset; // to build task_info struct
    struct ibv_mr mr_buffer_in, mr_buffer_out;
};

struct client_remote_ring {  
    // -- send over tcp --
    size_t N;
    struct ibv_mr mr_head, mr_tail, mr_mailbox; // will use rkey&addr

    clent_remote_ring() = default;
    explicit client_remote_ring(remote_ring const* ring) : N(ring->N), mr_head(ring->mr_head[0]), mr_tail(ring->mr_tail[0]), mr_mailbox(ring->mr_mailbox[0]) {}    
};

struct remote_ring { 
    // - server side only -
    size_t N;
    struct ibv_mr *mr_head, 
                  *mr_tail, 
                  *mr_mailbox;

    remote_ring(struct ibv_pd *pd, size_t *head, size_t *tail, void *mailbox, size_t n, size_t size_of_T) {
        N = n;
        mr_head = ibv_reg_mr(pd, head, sizeof(size_t), IBV_ACCESS_REMOTE_WRITE);
        mr_tail = ibv_reg_mr(pd, tail, sizeof(size_t), IBV_ACCESS_REMOTE_WRITE);
        mr_tail = ibv_reg_mr(pd, mr_mailbox, n*size_of_T, IBV_ACCESS_REMOTE_WRITE);
        if ( !(mr_head && mr_tail && mr_mailbox)) { 
            perror("failed ibv_reg_mr() for ring");
            exit(1);
        }
    }

    remote_ring(struct ibv_pd *pd, ring_buffer<task_info> *ring) : 
                remote_ring(pd, ring->_head, ring->_tail, ring->_mailbox, ring->N, sizeof(task_info)) {}

    remote_ring(struct ibv_pd *pd, ring_buffer<int> *ring) : 
                remote_ring(pd, ring->_head, ring->_tail, ring->_mailbox, ring->N, sizeof(int)) {}


    ~remote_ring() {
        ibv_dereg_mr(mr_head);
        ibv_dereg_mr(mr_tail);
        ibv_dereg_mr(mr_mailbox);
    }
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<image_processing_server> server;

    /* TODO: add memory region(s) for CPU-GPU queues */
    struct remote_ring ring_in, ring_out;

public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port), 
                                                        server(create_queues_server(THREAD_NUM)),
                                                        ring_in(pd, server->buffer_in),
                                                        ring_out(pd, server->buffer_out)
    {
        /* DONE: Initialize additional server MRs as needed. */

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        rings_packet msg = {
            .from_client_to_server = (client_remote_ring)ring_in, 
            .to_client_from_server = (client_remote_ring)ring_out,
            .server_img_in_offset  = images_in,
            .server_img_out_offset = images_out,
            .mr_buffer_in = mr_images_in,
            .mr_buffer_out = mr_images_out
        };
        send_over_socket(&msg, sizeof(msg));
    }

    ~server_queues_context()
    {
        // implicit destroys the remote_rings (only dereg the mr)    
        // at the end: implicit destroys the ~server() and terminate=1
    }

    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
                VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        //goto send_rdma_write;

                        post_rdma_write(
                            req->output_addr,                       // remote_dst
                            0,     // len
                            req->output_rkey,                       // rkey
                            0,                // local_src
                            mr_images_out->lkey,                    // lkey
                            dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                            (uint32_t*)&req->request_id);           // immediate
                    }
                    break;

                case IBV_WC_RDMA_WRITE: // just get the competion on the empty write
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

                    if (terminate)
                        got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
        } // while
        server.buffer_in->terminate[0]=1; // will close the rings & gpu, will happen also in destructor
        return; // will call destructor on exit from main
    }
};

class client_queues_context : public rdma_client_context {
private:

    int id_arr[OUTSTANDING_REQUESTS];
    uchar* dst_arr[OUTSTANDING_REQUESTS];
    int next_job;

    // remote context
    client_remote_ring ring_out, ring_in; // the MR of the server - named opposite of all else in-out
    uchar *server_img_in_offset, *server_img_out_offset; // to build task_info struct - can use the mr.addr but nevermind already
    struct ibv_mr mr_buffer_img_in, mr_buffer_img_out; // on the server

    // local regions
    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    

    // result of reads go here:
    task_info task_out; // only write source in here
    int finished_id;
    struct ibv_mr *mr_task_out, *mr_finished_id;

    size_t head_result, tail_result; // reuse for both rings
    struct ibv_mr *mr_head_result, *mr_tail_result;


    

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port), id_arr{0}, dst_arr{0}, next_job(0)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        rings_packet msg;
        recv_over_socket(&msg, sizeof(msg));
        ring_out = msg.from_client_to_server;
        ring_in = msg.to_client_from_server;
        server_img_in_offset = msg.server_img_in_offset;
        server_img_out_offset = msg.server_img_out_offset;
        mr_buffer_img_in = msg.mr_buffer_in;
        mr_buffer_img_out = msg.mr_buffer_out;

        // allocate mr for ring results
        mr_head_result = ibv_reg_mr(pd, &head_result, sizeof(size_t), IBV_ACCESS_LOCAL_WRITE);
        mr_tail_result = ibv_reg_mr(pd, &tail_result, sizeof(size_t), IBV_ACCESS_LOCAL_WRITE);
        mr_task_out    = ibv_reg_mr(pd, &task_out, sizeof(task_info), IBV_ACCESS_LOCAL_WRITE);//will read only
        mr_finished_id = ibv_reg_mr(pd, &finished_id, sizeof(int),    IBV_ACCESS_LOCAL_WRITE);
        if ( !(mr_head_result && mr_tail_result && mr_task_out && mr_finished_id)) { 
            perror("failed ibv_reg_mr() for local client side");
            exit(1);
        }
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
        // --- terminate server and wait to the write_imm ---
        
        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        int img_id = -1;
        uchar *img_in = nullptr;
        uchar *img_out = nullptr;

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[0]; // only one msg
        req->request_id = img_id;
        req->input_rkey = 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey =  0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }
        poll_cq(2); // also wait for write_imm
        // --- dereg ---
        ibv_dereg_mr(mr_head_result);
        ibv_dereg_mr(mr_tail_result);
        ibv_dereg_mr(mr_task_out);
        ibv_dereg_mr(mr_finished_id);
        ibv_dereg_mr(mr_images_in);
        ibv_dereg_mr(mr_images_out);
    }

    void poll_cq(int n) {
        int pass =0;
        struct ibv_wc wc;
        while (pass < n) {
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
                VERBS_WC_CHECK(wc);
                pass += ncqes;
            }
        }
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, 0); // only local read
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }


    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */

        // SKIP (already done):  create local read img_in mr (img_out will be in server untill the dequeue stage)
//
        // rdma write the img_in to the server's one (how to map?) 
        post_rdma_write(mr_buffer_img_in.addr + IMG_SZ*next_job, IMG_SZ*sizeof(uchar), mr_buffer_img_in.rkey,
                        img_in, mr_images_in->lkey, next_job,0);
        // create task_info object - will be with the remote pointers - use indecies instead?
        task_out = { 
            server_img_in_offset + IMG_SZ*next_job,
            server_img_out_offset + IMG_SZ*next_job,
            img_id
        };
        // push to ring: check not empty, write to mailbox, then increase the pointer TODO consider atomics after all else works
        post_rdma_read(&tail_result, sizeof(size_t), mr_tail_result->lkey, ring_out.mr_tail.addr, ring_out.mr_tail.rkey, next_job); 
        //do {
        post_rdma_read(&head_result, sizeof(size_t), mr_head_result->lkey, ring_out.mr_head.addr, ring_out.mr_head.rkey, next_job); 
        poll_cq(3);
        //} while (tail_result - head_result == OUTSTANDING_REQUESTS);
        if(tail_result - head_result == OUTSTANDING_REQUESTS)
            return false; // do not increase job number

        post_rdma_write(ring_out.mr_mailbox.addr + sizeof(task_info)*tail_result, sizeof(task_info), ring_out.mr_mailbox.rkey,
                        &task_out, mr_task_out->lkey, next_job, 0); // TODO different wr_id ?
        post_rdma_write(ring_out.mr_tail.addr, sizeof(size_t), ring_out.mr_tail.rkey,
                        &tail_result, mr_tail_result->lkey, next_job, 0);
        poll_cq(2);

        // return true if all writes are successful DONE check how to check for success, TODO wait for cqe?

        // ! - keep a id->local_img_out* mapping
        id_arr[next_job]  = img_id;
        dst_arr[next_job] = img_out;
        next_job = (next_job + 1) % OUTSTANDING_REQUESTS;
        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */

        // get a finished id (all the steps in pop)
        post_rdma_read(&head_result, sizeof(size_t), mr_head_result->lkey, ring_in.mr_head.addr, ring_in.mr_head.rkey, 0); 
        post_rdma_read(&tail_result, sizeof(size_t), mr_tail_result->lkey, ring_in.mr_tail.addr, ring_in.mr_tail.rkey, 0); 
        poll_cq(2);
        if(tail_result == head_result)
            return false; // do not increase job number
        post_rdma_read(&finished_id, sizeof(int), mr_task_in->lkey, ring_in.mr_mailbox.addr + sizeof(int)*head_result,
                ring_in.mr_mailbox.rkey, 0); // TODO different wr_id ?
        post_rdma_write(ring_in.mr_head.addr, sizeof(size_t), ring_in.mr_head.rkey,
                        &head_result, mr_head_result->lkey, 0, 0);
        poll_cq(2);

        // find img_out*
        uchar *img_out;
        int job_index;
        for(int i=0; i<OUTSTANDING_REQUESTS; i++) 
            if (id_arr[i] == finished_id) {
                img_out = dst_arr[i];
                job_index = i;
                break;
            }
        // read the img_out to the local one
        post_rdma_read(img_out, IMG_SZ*sizeof(uchar), mr_images_out->lkey,
                        mr_buffer_img_out.addr + IMG_SZ*job_index, mr_buffer_img_out.rkey, 0);
        poll_cq(1);
        // mark success
        img_id[0] = finished_id;
        return true;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
