/*
 * dct_test_gpu.c — DC write test with GPU memory registered via DMA-BUF.
 *
 * Replaces ibv_reg_mr(host) with:
 *   cudaMalloc → cuMemGetHandleForAddressRange(DMA_BUF_FD) → ibv_reg_mr_ex
 *
 * DMA-BUF registration follows the perftest pattern (cuda_memory.c):
 *   - alloc rounded up to 64KB GPU page
 *   - export from 4KB-page-aligned address
 *   - iova = original cudaMalloc ptr, fd_offset = small page gap
 *   - RDMA addresses the cudaMalloc ptr directly (no +64KB offset)
 *
 * Usage:
 *   loopback: ./dct_test_gpu -l [-d mlx5_1] [-g 0]
 *   server:   ./dct_test_gpu -s [-p port] [-d mlx5_1] [-g 0]
 *   client:   ./dct_test_gpu -c -a <ip> [-p port] [-d mlx5_1] [-g 0]
 *
 * Build:
 *   gcc -O2 -o dct_test_gpu dct_test_gpu.c -libverbs -lmlx5 \
 *       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <getopt.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#define CQ_DEPTH     128
#define BUF_SIZE     4096
#define POLL_RETRIES 100000
#define GID_IDX      3
#define TCP_PORT     19997
#define ACCEL_PAGE_SIZE (64 * 1024)  /* GPU page / BAR chunk size (perftest) */

struct dct_remote_info {
    uint32_t dct_num;
    uint32_t rkey;
    uint64_t addr;
    uint64_t dct_key;
    uint8_t  gid[16];
    uint8_t  port_num;
};

struct gpu_mr {
    void        *dptr;
    struct ibv_mr *mr;
    size_t       size;
};

static int g_gpu = 0;
static int g_host_recv = 0; /* server: host MR for DCT buffer */
static int g_host_send = 0; /* client: host MR for DCI source buffer */
static int g_iters = 1;     /* client: number of RDMA writes */
static int g_no_sync = 0;   /* client: skip cudaDeviceSynchronize before post */
static size_t g_payload = 64; /* bytes per transfer */

static void gpu_check(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

static void cu_check(CUresult r, const char *msg)
{
    if (r != CUDA_SUCCESS) {
        const char *s = NULL;
        cuGetErrorString(r, &s);
        fprintf(stderr, "CUDA driver error %s: %s\n", msg, s ? s : "?");
        exit(1);
    }
}

static int gpu_init(void)
{
    cu_check(cuInit(0), "cuInit");
    int ndev = 0;
    gpu_check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices\n");
        return -1;
    }
    if (g_gpu >= ndev) g_gpu = 0;
    gpu_check(cudaSetDevice(g_gpu), "cudaSetDevice");
    int dma_ok = 0;
    CUdevice cu_dev;
    cu_check(cuDeviceGet(&cu_dev, g_gpu), "cuDeviceGet");
    cu_check(cuDeviceGetAttribute(&dma_ok, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                  cu_dev),
             "DMA_BUF_SUPPORTED");
    if (!dma_ok) {
        fprintf(stderr, "GPU %d does not support DMA-BUF\n", g_gpu);
        return -1;
    }
    printf("[gpu] using GPU %d\n", g_gpu);
    return 0;
}

static int gpu_reg_mr(struct ibv_pd *pd, struct gpu_mr *gm, size_t size, int access)
{
    /* perftest-aligned DMA-BUF registration:
     *   - round up allocation to GPU page (64KB)
     *   - round down export address to host page (4KB)
     *   - iova = original cudaMalloc ptr; fd_offset = small page gap
     *   - RDMA uses the original cudaMalloc ptr directly (no +64KB hack)
     */
    long host_page = sysconf(_SC_PAGESIZE);
    if (host_page <= 0) host_page = 4096;

    size_t alloc_size = (size + ACCEL_PAGE_SIZE - 1) & ~((size_t)ACCEL_PAGE_SIZE - 1);

    gm->size = size;
    gm->dptr = NULL;
    gm->mr = NULL;

    gpu_check(cudaMalloc(&gm->dptr, alloc_size), "cudaMalloc");

    uintptr_t aligned_ptr = (uintptr_t)gm->dptr & ~((uintptr_t)host_page - 1);
    uint64_t fd_offset = (uint64_t)(uintptr_t)gm->dptr - (uint64_t)aligned_ptr;
    size_t aligned_size = (size + fd_offset + (size_t)host_page - 1) & ~((size_t)host_page - 1);

    printf("[gpu] dptr=%p alloc=%zu aligned_ptr=0x%lx fd_offset=%lu aligned_size=%zu\n",
           gm->dptr, alloc_size, (unsigned long)aligned_ptr,
           (unsigned long)fd_offset, aligned_size);

    int fd = -1;
    CUresult cr = cuMemGetHandleForAddressRange(
        &fd, (CUdeviceptr)aligned_ptr, aligned_size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (cr != CUDA_SUCCESS) {
        cr = cuMemGetHandleForAddressRange(
            &fd, (CUdeviceptr)aligned_ptr, aligned_size,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    }
    cu_check(cr, "cuMemGetHandleForAddressRange");

    uint64_t iova = (uint64_t)(uintptr_t)gm->dptr;
    struct ibv_mr_init_attr attr = {
        .length = aligned_size,
        .access = access,
        .comp_mask = IBV_REG_MR_MASK_IOVA | IBV_REG_MR_MASK_FD |
                     IBV_REG_MR_MASK_FD_OFFSET,
        .iova = iova,
        .fd = fd,
        .fd_offset = fd_offset,
    };
    gm->mr = ibv_reg_mr_ex(pd, &attr);
    close(fd);
    if (!gm->mr) {
        /* fallback to ibv_reg_dmabuf_mr */
        fd = -1;
        cr = cuMemGetHandleForAddressRange(
            &fd, (CUdeviceptr)aligned_ptr, aligned_size,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        cu_check(cr, "cuMemGetHandleForAddressRange retry");
        gm->mr = ibv_reg_dmabuf_mr(pd, fd_offset, aligned_size, iova, fd, access);
        close(fd);
    }
    if (!gm->mr) {
        fprintf(stderr, "GPU MR register failed: %s\n", strerror(errno));
        cudaFree(gm->dptr);
        gm->dptr = NULL;
        return -1;
    }
    printf("[gpu] dmabuf MR: dptr=%p lkey=0x%x rkey=0x%x iova=0x%lx fd_offset=%lu\n",
           gm->dptr, gm->mr->lkey, gm->mr->rkey,
           (unsigned long)iova, (unsigned long)fd_offset);
    return 0;
}

static void gpu_dereg_mr(struct gpu_mr *gm)
{
    if (gm->mr) ibv_dereg_mr(gm->mr);
    if (gm->dptr) cudaFree(gm->dptr);
    memset(gm, 0, sizeof(*gm));
}

static int find_active_roce(struct ibv_context **out_ctx, uint8_t *out_port,
                            char *dev_name, size_t n, const char *want_dev)
{
    int ndev;
    struct ibv_device **list = ibv_get_device_list(&ndev);
    if (!list || ndev == 0) return -1;

    for (int i = 0; i < ndev; i++) {
        const char *name = ibv_get_device_name(list[i]);
        if (want_dev && want_dev[0] && strcmp(name, want_dev) != 0)
            continue;
        struct ibv_context *ctx = ibv_open_device(list[i]);
        if (!ctx) continue;
        struct ibv_device_attr da;
        if (ibv_query_device(ctx, &da)) { ibv_close_device(ctx); continue; }
        for (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
            struct ibv_port_attr pa;
            if (ibv_query_port(ctx, p, &pa)) continue;
            if (pa.state == IBV_PORT_ACTIVE &&
                pa.link_layer == IBV_LINK_LAYER_ETHERNET) {
                *out_ctx = ctx;
                *out_port = p;
                snprintf(dev_name, n, "%s", name);
                ibv_free_device_list(list);
                return 0;
            }
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(list);
    return -1;
}

static int create_dct(struct ibv_context *ctx, struct ibv_pd *pd,
                      struct ibv_cq *scq, struct ibv_cq *rcq,
                      struct ibv_srq *srq, uint8_t port_num,
                      union ibv_gid gid, uint64_t dct_key,
                      struct ibv_qp **out_dct)
{
    struct ibv_qp_init_attr_ex attr_ex = {};
    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = scq;
    attr_ex.recv_cq = rcq;
    attr_ex.srq = srq;
    attr_ex.cap.max_send_wr = 1;
    attr_ex.cap.max_recv_wr = 64;
    attr_ex.cap.max_send_sge = 1;
    attr_ex.cap.max_recv_sge = 1;
    attr_ex.pd = pd;
    attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD;

    struct mlx5dv_qp_init_attr attr_dv = {};
    attr_dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    attr_dv.dc_init_attr.dct_access_key = dct_key;

    struct ibv_qp *dct = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!dct) return -1;

    struct ibv_port_attr pa;
    ibv_query_port(ctx, port_num, &pa);

    struct ibv_qp_attr qpa = {};
    qpa.qp_state = IBV_QPS_INIT;
    qpa.pkey_index = 0;
    qpa.port_num = port_num;
    qpa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_ATOMIC;
    if (ibv_modify_qp(dct, &qpa,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS)) {
        ibv_destroy_qp(dct);
        return -1;
    }

    memset(&qpa, 0, sizeof(qpa));
    qpa.qp_state = IBV_QPS_RTR;
    qpa.path_mtu = pa.active_mtu;
    qpa.min_rnr_timer = 12;
    qpa.ah_attr.is_global = 1;
    qpa.ah_attr.port_num = port_num;
    qpa.ah_attr.grh.dgid = gid;
    qpa.ah_attr.grh.sgid_index = GID_IDX;
    qpa.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(dct, &qpa,
                      IBV_QP_STATE | IBV_QP_MIN_RNR_TIMER | IBV_QP_PATH_MTU |
                          IBV_QP_AV)) {
        ibv_destroy_qp(dct);
        return -1;
    }
    *out_dct = dct;
    return 0;
}

static int create_dci(struct ibv_context *ctx, struct ibv_pd *pd,
                      struct ibv_cq *scq, struct ibv_cq *rcq,
                      uint8_t port_num, union ibv_gid gid,
                      struct ibv_qp **out_dci)
{
    struct ibv_qp_init_attr_ex attr_ex = {};
    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = scq;
    attr_ex.recv_cq = rcq;
    attr_ex.cap.max_send_wr = 128;
    attr_ex.cap.max_recv_wr = 0;
    attr_ex.cap.max_send_sge = 2;
    attr_ex.cap.max_recv_sge = 0;
    attr_ex.pd = pd;
    attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;

    struct mlx5dv_qp_init_attr attr_dv = {};
    attr_dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC |
                        MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    attr_dv.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

    struct ibv_qp *dci = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!dci) return -1;

    struct ibv_port_attr pa;
    ibv_query_port(ctx, port_num, &pa);

    struct ibv_qp_attr qpa = {};
    qpa.qp_state = IBV_QPS_INIT;
    qpa.pkey_index = 0;
    qpa.port_num = port_num;
    if (ibv_modify_qp(dci, &qpa,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT)) {
        ibv_destroy_qp(dci);
        return -1;
    }

    memset(&qpa, 0, sizeof(qpa));
    qpa.qp_state = IBV_QPS_RTR;
    qpa.path_mtu = pa.active_mtu;
    qpa.ah_attr.is_global = 1;
    qpa.ah_attr.port_num = port_num;
    qpa.ah_attr.grh.dgid = gid;
    qpa.ah_attr.grh.sgid_index = GID_IDX;
    qpa.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(dci, &qpa, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV)) {
        ibv_destroy_qp(dci);
        return -1;
    }

    memset(&qpa, 0, sizeof(qpa));
    qpa.qp_state = IBV_QPS_RTS;
    qpa.sq_psn = 0;
    qpa.timeout = 14;
    qpa.retry_cnt = 7;
    qpa.rnr_retry = 7;
    qpa.max_rd_atomic = 1;
    if (ibv_modify_qp(dci, &qpa,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_MAX_QP_RD_ATOMIC)) {
        ibv_destroy_qp(dci);
        return -1;
    }
    *out_dci = dci;
    return 0;
}

static int poll_send_cq(struct ibv_cq *cq)
{
    struct ibv_wc wc;
    for (int i = 0; i < POLL_RETRIES; i++) {
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) return -1;
        if (n == 0) continue;
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "CQ error status=%d vendor=0x%x\n", wc.status,
                    wc.vendor_err);
            return -1;
        }
        printf("CQE OK opcode=%d wr_id=%lu byte_len=%u\n", wc.opcode, wc.wr_id,
               wc.byte_len);
        return 0;
    }
    fprintf(stderr, "CQ poll timeout\n");
    return -1;
}

static int dc_write(struct ibv_qp *dci, struct ibv_pd *pd, struct ibv_ah *ah,
                    uint32_t remote_dctn, uint64_t dct_key,
                    struct gpu_mr *local, struct gpu_mr *remote, size_t len)
{
    struct ibv_qp_ex *qpx = ibv_qp_to_qp_ex(dci);
    struct mlx5dv_qp_ex *dv = mlx5dv_qp_ex_from_ibv_qp_ex(qpx);
    uint64_t raddr = (uint64_t)(uintptr_t)remote->dptr;
    uint64_t laddr = (uint64_t)(uintptr_t)local->dptr;

    ibv_wr_start(qpx);
    qpx->wr_id = 1;
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qpx, remote->mr->rkey, raddr);
    mlx5dv_wr_set_dc_addr(dv, ah, remote_dctn, dct_key);
    ibv_wr_set_sge(qpx, local->mr->lkey, laddr, len);
    if (ibv_wr_complete(qpx)) {
        fprintf(stderr, "ibv_wr_complete failed\n");
        return -1;
    }
    return 0;
}

static int run_loopback(const char *want_dev)
{
    struct ibv_context *ctx = NULL;
    struct ibv_pd *pd = NULL;
    struct ibv_cq *scq = NULL, *rcq = NULL;
    struct ibv_srq *srq = NULL;
    struct ibv_qp *dct = NULL, *dci = NULL;
    struct ibv_ah *ah = NULL;
    struct gpu_mr dct_gm = {}, dci_gm = {};
    uint8_t port_num = 0;
    char dev_name[64];
    union ibv_gid gid;
    int rc = 1;

    if (gpu_init() != 0) return 1;
    if (find_active_roce(&ctx, &port_num, dev_name, sizeof(dev_name), want_dev))
        return 1;
    printf("[loopback] %s port %u\n", dev_name, port_num);

    ibv_query_gid(ctx, port_num, GID_IDX, &gid);

    pd = ibv_alloc_pd(ctx);
    scq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    rcq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    if (!pd || !scq || !rcq) goto cleanup;

    struct ibv_srq_init_attr_ex sa = {};
    sa.comp_mask = IBV_SRQ_INIT_ATTR_TYPE | IBV_SRQ_INIT_ATTR_PD;
    sa.attr.max_wr = 64;
    sa.attr.max_sge = 1;
    sa.pd = pd;
    sa.srq_type = IBV_SRQT_BASIC;
    srq = ibv_create_srq_ex(ctx, &sa);
    if (!srq) goto cleanup;

    srand((unsigned)(time(NULL) ^ getpid()));
    uint64_t dct_key = ((uint64_t)rand() << 32) | (uint64_t)rand();

    if (create_dct(ctx, pd, scq, rcq, srq, port_num, gid, dct_key, &dct) ||
        create_dci(ctx, pd, scq, rcq, port_num, gid, &dci))
        goto cleanup;

    if (gpu_reg_mr(pd, &dct_gm, BUF_SIZE,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ) ||
        gpu_reg_mr(pd, &dci_gm, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE))
        goto cleanup;

    const size_t len = 64;
    gpu_check(cudaMemset((char *)dci_gm.dptr, 0xAB, len),
              "cudaMemset src");
    gpu_check(cudaMemset((char *)dct_gm.dptr, 0, len),
              "cudaMemset dst");

    struct ibv_ah_attr ah_attr = {};
    ah_attr.is_global = 1;
    ah_attr.port_num = port_num;
    ah_attr.grh.dgid = gid;
    ah_attr.grh.sgid_index = GID_IDX;
    ah_attr.grh.hop_limit = 64;
    ah = ibv_create_ah(pd, &ah_attr);
    if (!ah) goto cleanup;

    printf("RDMA Write: DCI#%u -> DCT#%u, %zu bytes (GPU dmabuf)\n", dci->qp_num,
           dct->qp_num, len);
    if (dc_write(dci, pd, ah, dct->qp_num, dct_key, &dci_gm, &dct_gm, len) ||
        poll_send_cq(scq))
        goto cleanup;

    gpu_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned char host[64], expect[64];
    memset(expect, 0xAB, len);
    gpu_check(cudaMemcpy(host, (char *)dct_gm.dptr, len,
                         cudaMemcpyDeviceToHost),
              "D2H verify");
    if (memcmp(host, expect, len) != 0) {
        fprintf(stderr, "GPU data mismatch, first 16 bytes got:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %02x", host[i]);
        fprintf(stderr, "\n");
        goto cleanup;
    }
    printf("GPU buffer verified OK (%zu bytes 0xAB)\n", len);
    printf("\n*** Loopback PASSED (GPU dmabuf) ***\n");
    rc = 0;

cleanup:
    if (ah) ibv_destroy_ah(ah);
    gpu_dereg_mr(&dci_gm);
    gpu_dereg_mr(&dct_gm);
    if (dci) ibv_destroy_qp(dci);
    if (dct) ibv_destroy_qp(dct);
    if (srq) ibv_destroy_srq(srq);
    if (rcq) ibv_destroy_cq(rcq);
    if (scq) ibv_destroy_cq(scq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    return rc;
}

static int run_server(int tcp_port, const char *want_dev)
{
    struct ibv_context *ctx = NULL;
    struct ibv_pd *pd = NULL;
    struct ibv_cq *scq = NULL, *rcq = NULL;
    struct ibv_srq *srq = NULL;
    struct ibv_qp *dct = NULL;
    struct gpu_mr dct_gm = {};
    void *host_buf = NULL;
    struct ibv_mr *host_mr = NULL;
    uint8_t port_num = 0;
    char dev_name[64];
    union ibv_gid gid;
    int sock = -1, client_fd = -1;
    int rc = 1;

    if (gpu_init() != 0) return 1;
    if (find_active_roce(&ctx, &port_num, dev_name, sizeof(dev_name), want_dev))
        return 1;
    printf("[server] %s port %u\n", dev_name, port_num);
    ibv_query_gid(ctx, port_num, GID_IDX, &gid);

    pd = ibv_alloc_pd(ctx);
    scq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    rcq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    if (!pd || !scq || !rcq) goto cleanup;

    struct ibv_srq_init_attr_ex sa = {};
    sa.comp_mask = IBV_SRQ_INIT_ATTR_TYPE | IBV_SRQ_INIT_ATTR_PD;
    sa.attr.max_wr = 64;
    sa.attr.max_sge = 1;
    sa.pd = pd;
    sa.srq_type = IBV_SRQT_BASIC;
    srq = ibv_create_srq_ex(ctx, &sa);
    if (!srq) goto cleanup;

    srand((unsigned)(time(NULL) ^ getpid()));
    uint64_t dct_key = ((uint64_t)rand() << 32) | (uint64_t)rand();

    if (create_dct(ctx, pd, scq, rcq, srq, port_num, gid, dct_key, &dct))
        goto cleanup;

    if (g_host_recv) {
        host_buf = calloc(1, BUF_SIZE);
        if (!host_buf) goto cleanup;
        host_mr = ibv_reg_mr(pd, host_buf, BUF_SIZE,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ);
        if (!host_mr) goto cleanup;
        printf("[server] host recv MR buf=%p rkey=0x%x\n", host_buf, host_mr->rkey);
    } else if (gpu_reg_mr(pd, &dct_gm, BUF_SIZE,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ)) {
        goto cleanup;
    }
    if (!g_host_recv)
        gpu_check(cudaMemset((char *)dct_gm.dptr, 0, BUF_SIZE),
                  "cudaMemset dst");

    printf("[server] DCT#=%u key=0x%016lx (GPU recv buffer)\n", dct->qp_num,
           dct_key);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(tcp_port);
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(sock, 1) < 0)
        goto cleanup;

    printf("[server] waiting TCP %d...\n", tcp_port);
    client_fd = accept(sock, NULL, NULL);
    if (client_fd < 0) goto cleanup;

    struct dct_remote_info info = {};
    info.dct_num = dct->qp_num;
    if (g_host_recv) {
        info.rkey = host_mr->rkey;
        info.addr = (uint64_t)(uintptr_t)host_buf;
    } else {
        info.rkey = dct_gm.mr->rkey;
        info.addr = (uint64_t)(uintptr_t)dct_gm.dptr;
    }
    info.dct_key = dct_key;
    memcpy(info.gid, gid.raw, 16);
    info.port_num = port_num;
    if (write(client_fd, &info, sizeof(info)) != (ssize_t)sizeof(info))
        goto cleanup;

    if (g_iters > 1) {
        int32_t cfg_iters = htonl(g_iters);
        if (write(client_fd, &cfg_iters, sizeof(cfg_iters)) != (ssize_t)sizeof(cfg_iters))
            goto cleanup;
    }

    int verify_fail = 0;
    int last_seq = -1;

    if (g_iters > 1) {
        printf("[server] per-iter verify mode, %d iterations\n", g_iters);
        char *recv_host = malloc(g_payload + 1);
        if (!recv_host) goto cleanup;

        for (int i = 0; i < g_iters; i++) {
            int32_t seq_net;
            if (read(client_fd, &seq_net, sizeof(seq_net)) != (ssize_t)sizeof(seq_net)) {
                free(recv_host);
                goto cleanup;
            }
            int seq = ntohl(seq_net);
            last_seq = seq;

            if (!g_host_recv)
                gpu_check(cudaDeviceSynchronize(), "server sync before verify");

            if (g_host_recv) {
                memcpy(recv_host, host_buf, g_payload);
            } else {
                gpu_check(cudaMemcpy(recv_host, (char *)dct_gm.dptr,
                                     g_payload, cudaMemcpyDeviceToHost),
                          "D2H verify");
            }
            recv_host[g_payload] = '\0';

            int expect = seq;
            int got = -1;
            if (sscanf(recv_host, "SEQ=%d", &got) != 1 || got != expect) {
                fprintf(stderr, "[server] iter %d FAIL: expect SEQ=%d got \"%.*s\"\n",
                        i, expect, (int)g_payload, recv_host);
                verify_fail++;
            }

            uint8_t ack = verify_fail ? 1 : 0;
            if (write(client_fd, &ack, 1) != 1) {
                free(recv_host);
                goto cleanup;
            }
        }
        free(recv_host);

        char done[4];
        read(client_fd, done, sizeof(done));
        printf("[server] verify_fail=%d last_seq=%d\n", verify_fail, last_seq);
        if (verify_fail == 0)
            printf("\n*** Server PASSED (%d iters, GPU dmabuf) ***\n", g_iters);
        else
            fprintf(stderr, "\n*** Server FAILED (%d/%d verify errors) ***\n",
                    verify_fail, g_iters);
        rc = verify_fail ? 1 : 0;
        goto cleanup;
    }

    char done[4];
    read(client_fd, done, sizeof(done));

    gpu_check(cudaDeviceSynchronize(), "server cudaDeviceSynchronize");

    char host[256];
    if (g_host_recv) {
        memcpy(host, host_buf, sizeof(host) - 1);
    } else {
        gpu_check(cudaMemcpy(host, (char *)dct_gm.dptr,
                             sizeof(host) - 1, cudaMemcpyDeviceToHost),
                  "D2H");
    }
    host[255] = '\0';
    printf("[server] GPU buffer: \"%s\"\n", host);
    if (host[0] == '\0') {
        fprintf(stderr, "[server] FAIL: GPU buffer empty after RDMA write\n");
        rc = 1;
        goto cleanup;
    }
    printf("\n*** Server PASSED (GPU dmabuf) ***\n");
    rc = 0;

cleanup:
    if (client_fd >= 0) close(client_fd);
    if (sock >= 0) close(sock);
    if (host_mr) ibv_dereg_mr(host_mr);
    free(host_buf);
    gpu_dereg_mr(&dct_gm);
    if (dct) ibv_destroy_qp(dct);
    if (srq) ibv_destroy_srq(srq);
    if (rcq) ibv_destroy_cq(rcq);
    if (scq) ibv_destroy_cq(scq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    return rc;
}

static int run_client(const char *server_ip, int tcp_port, const char *want_dev)
{
    struct ibv_context *ctx = NULL;
    struct ibv_pd *pd = NULL;
    struct ibv_cq *scq = NULL, *rcq = NULL;
    struct ibv_qp *dci = NULL;
    struct ibv_ah *ah = NULL;
    struct gpu_mr dci_gm = {};
    void *host_send = NULL;
    struct ibv_mr *host_send_mr = NULL;
    uint8_t port_num = 0;
    char dev_name[64];
    union ibv_gid local_gid;
    int sock = -1;
    int rc = 1;

    if (g_host_send) {
        cu_check(cuInit(0), "cuInit");
    } else if (gpu_init() != 0) {
        return 1;
    }
    if (find_active_roce(&ctx, &port_num, dev_name, sizeof(dev_name), want_dev))
        return 1;
    printf("[client] %s port %u\n", dev_name, port_num);
    ibv_query_gid(ctx, port_num, GID_IDX, &local_gid);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);
    inet_pton(AF_INET, server_ip, &addr.sin_addr);
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        goto cleanup;
    }

    struct dct_remote_info info;
    if (read(sock, &info, sizeof(info)) != (ssize_t)sizeof(info))
        goto cleanup;
    printf("[client] remote DCT#=%u rkey=0x%x addr=%lx\n", info.dct_num,
           info.rkey, info.addr);

    pd = ibv_alloc_pd(ctx);
    scq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    rcq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    if (!pd || !scq || !rcq) goto cleanup;

    if (create_dci(ctx, pd, scq, rcq, port_num, local_gid, &dci))
        goto cleanup;

    char msg[256];
    snprintf(msg, sizeof(msg), "GPU dmabuf DC write pid=%d dev=%s", getpid(),
             dev_name);
    size_t len = strlen(msg) + 1;
    (void)len;

    uint32_t lkey;
    uint64_t sge_addr;
    if (g_host_send) {
        host_send = malloc(BUF_SIZE);
        if (!host_send) goto cleanup;
        memcpy(host_send, msg, len);
        host_send_mr = ibv_reg_mr(pd, host_send, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
        if (!host_send_mr) goto cleanup;
        lkey = host_send_mr->lkey;
        sge_addr = (uint64_t)(uintptr_t)host_send;
        printf("[client] host send MR buf=%p lkey=0x%x\n", host_send, lkey);
    } else {
        if (gpu_reg_mr(pd, &dci_gm, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE))
            goto cleanup;
        gpu_check(cudaMemcpy((char *)dci_gm.dptr, msg, len,
                             cudaMemcpyHostToDevice),
                  "H2D src");
        gpu_check(cudaDeviceSynchronize(), "pre-post sync");
        lkey = dci_gm.mr->lkey;
        sge_addr = (uint64_t)(uintptr_t)dci_gm.dptr;
    }

    struct ibv_ah_attr ah_attr = {};
    ah_attr.is_global = 1;
    ah_attr.port_num = port_num;
    ah_attr.grh.sgid_index = GID_IDX;
    ah_attr.grh.hop_limit = 64;
    memcpy(ah_attr.grh.dgid.raw, info.gid, 16);
    ah = ibv_create_ah(pd, &ah_attr);
    if (!ah) goto cleanup;

    struct ibv_qp_ex *qpx = ibv_qp_to_qp_ex(dci);
    struct mlx5dv_qp_ex *dv = mlx5dv_qp_ex_from_ibv_qp_ex(qpx);

    if (g_iters > 1) {
        int32_t cfg_iters;
        if (read(sock, &cfg_iters, sizeof(cfg_iters)) != (ssize_t)sizeof(cfg_iters))
            goto cleanup;
        if (ntohl(cfg_iters) != g_iters) {
            fprintf(stderr, "[client] server expects %d iters, we have %d\n",
                    ntohl(cfg_iters), g_iters);
            goto cleanup;
        }
    }

    char *payload = malloc(g_payload);
    if (!payload) goto cleanup;

    int cq_fail = 0;
    printf("[client] stress: iters=%d payload=%zu sync=%s\n",
           g_iters, g_payload, g_no_sync ? "OFF" : "ON");

    for (int i = 0; i < g_iters; i++) {
        snprintf(payload, g_payload, "SEQ=%d pid=%d dev=%s pad", i, getpid(),
                 dev_name);

        if (g_host_send) {
            memcpy(host_send, payload, g_payload);
        } else {
            gpu_check(cudaMemcpy((char *)dci_gm.dptr, payload,
                                 g_payload, cudaMemcpyHostToDevice),
                      "H2D src");
            if (!g_no_sync)
                gpu_check(cudaDeviceSynchronize(), "pre-post sync");
        }

        ibv_wr_start(qpx);
        qpx->wr_id = (uint64_t)(i + 1);
        qpx->wr_flags = IBV_SEND_SIGNALED;
        ibv_wr_rdma_write(qpx, info.rkey, info.addr);
        mlx5dv_wr_set_dc_addr(dv, ah, info.dct_num, info.dct_key);
        ibv_wr_set_sge(qpx, lkey, sge_addr, g_payload);
        if (ibv_wr_complete(qpx)) {
            fprintf(stderr, "[client] iter %d: ibv_wr_complete failed\n", i);
            cq_fail++;
            break;
        }
        if (poll_send_cq(scq)) {
            fprintf(stderr, "[client] iter %d: CQ error\n", i);
            cq_fail++;
            break;
        }

        if (g_iters > 1) {
            int32_t seq_net = htonl(i);
            if (write(sock, &seq_net, sizeof(seq_net)) != (ssize_t)sizeof(seq_net))
                goto cleanup;
            uint8_t ack;
            if (read(sock, &ack, 1) != 1 || ack != 0) {
                fprintf(stderr, "[client] iter %d: server verify NACK\n", i);
                cq_fail++;
            }
        }

        if (g_iters >= 100 && (i + 1) % 100 == 0)
            printf("[client] progress %d/%d\n", i + 1, g_iters);
    }

    free(payload);

    if (cq_fail) {
        fprintf(stderr, "[client] FAIL: errors=%d\n", cq_fail);
        goto cleanup;
    }

    write(sock, "DONE", 4);
    printf("\n*** Client PASSED (%d iters, GPU dmabuf, sync=%s) ***\n",
           g_iters, g_no_sync ? "OFF" : "ON");
    rc = 0;
    goto cleanup;

cleanup:
    if (sock >= 0) close(sock);
    if (ah) ibv_destroy_ah(ah);
    if (host_send_mr) ibv_dereg_mr(host_send_mr);
    free(host_send);
    gpu_dereg_mr(&dci_gm);
    if (dci) ibv_destroy_qp(dci);
    if (rcq) ibv_destroy_cq(rcq);
    if (scq) ibv_destroy_cq(scq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    return rc;
}

static void usage(const char *p)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s -l [-d dev] [-g gpu_id]          # loopback GPU dmabuf\n"
            "  %s -s [-p port] [-d dev] [-g gpu] [-H host recv MR]\n"
            "  %s -c -a <ip> [-p port] [-d dev] [-g gpu] [-S host send MR]\n"
            "      [-n iters] [-N no cuda sync] [-z payload_bytes]\n",
            p, p, p);
}

int main(int argc, char **argv)
{
    int loopback = 0, is_server = 0, is_client = 0, tcp_port = TCP_PORT;
    char *server_ip = NULL, *want_dev = NULL;
    int opt;

    while ((opt = getopt(argc, argv, "lsca:p:d:g:HSn:Nz:h")) != -1) {
        switch (opt) {
        case 'l': loopback = 1; break;
        case 's': is_server = 1; break;
        case 'c': is_client = 1; break;
        case 'a': server_ip = optarg; break;
        case 'p': tcp_port = atoi(optarg); break;
        case 'd': want_dev = optarg; break;
        case 'g': g_gpu = atoi(optarg); break;
        case 'H': g_host_recv = 1; break;
        case 'S': g_host_send = 1; break;
        case 'n': g_iters = atoi(optarg); break;
        case 'N': g_no_sync = 1; break;
        case 'z': g_payload = (size_t)atoi(optarg); break;
        default: usage(argv[0]); return 1;
        }
    }

    if (g_payload == 0 || g_payload > ACCEL_PAGE_SIZE) {
        fprintf(stderr, "payload must be 1..%d bytes\n",
                (int)ACCEL_PAGE_SIZE);
        return 1;
    }

    if (loopback) return run_loopback(want_dev);
    if (is_server) return run_server(tcp_port, want_dev);
    if (is_client) {
        if (!server_ip) { usage(argv[0]); return 1; }
        return run_client(server_ip, tcp_port, want_dev);
    }
    usage(argv[0]);
    return 1;
}
