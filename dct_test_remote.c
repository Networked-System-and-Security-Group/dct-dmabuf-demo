/*
 * dct_test_remote.c — split DCT(server) / DCI(client) DC write test.
 * Same perftest patterns as dct_test.c; TCP exchanges DCT metadata only.
 *
 *   Server (113): ./dct_test_remote -s [-p port] [-d mlx5_1]
 *   Client (112): ./dct_test_remote -c -a <server-ip> [-p port] [-d mlx5_1]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#define CQ_DEPTH     128
#define BUF_SIZE     4096
#define POLL_RETRIES 100000
#define GID_IDX      3
#define TCP_PORT     19998

struct dct_remote_info {
    uint32_t dct_num;
    uint32_t rkey;
    uint64_t addr;
    uint64_t dct_key;
    uint8_t  gid[16];
    uint8_t  port_num;
};

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

static int run_server(int tcp_port, const char *want_dev)
{
    struct ibv_context *ctx = NULL;
    struct ibv_pd *pd = NULL;
    struct ibv_cq *scq = NULL, *rcq = NULL;
    struct ibv_srq *srq = NULL;
    struct ibv_qp *dct = NULL;
    struct ibv_mr *mr = NULL;
    void *buf = NULL;
    uint8_t port_num = 0;
    char dev_name[64];
    union ibv_gid gid;
    int sock = -1, client_fd = -1;
    int rc = 1;

    srand((unsigned)(time(NULL) ^ getpid()));
    uint64_t dct_key = ((uint64_t)rand() << 32) | (uint64_t)rand();

    if (find_active_roce(&ctx, &port_num, dev_name, sizeof(dev_name), want_dev)) {
        fprintf(stderr, "No active RoCE port\n");
        return 1;
    }
    printf("[server] %s port %u\n", dev_name, port_num);

    if (ibv_query_gid(ctx, port_num, GID_IDX, &gid)) {
        fprintf(stderr, "ibv_query_gid idx=%d failed\n", GID_IDX);
        goto cleanup;
    }

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

    dct = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!dct) goto cleanup;

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
                          IBV_QP_ACCESS_FLAGS))
        goto cleanup;

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
                          IBV_QP_AV))
        goto cleanup;

    buf = calloc(1, BUF_SIZE);
    mr = ibv_reg_mr(pd, buf, BUF_SIZE,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ);
    if (!buf || !mr) goto cleanup;

    printf("[server] DCT#=%u key=0x%016lx rkey=0x%x buf=%p gid_idx=%d\n",
           dct->qp_num, dct_key, mr->rkey, buf, GID_IDX);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) goto cleanup;
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
    info.rkey = mr->rkey;
    info.addr = (uint64_t)(uintptr_t)buf;
    info.dct_key = dct_key;
    memcpy(info.gid, gid.raw, 16);
    info.port_num = port_num;
    if (write(client_fd, &info, sizeof(info)) != (ssize_t)sizeof(info))
        goto cleanup;

    char done[4];
    read(client_fd, done, sizeof(done));
    printf("[server] buffer: \"%.64s\"\n", (char *)buf);
    printf("\n*** Server PASSED ***\n");
    rc = 0;

cleanup:
    if (client_fd >= 0) close(client_fd);
    if (sock >= 0) close(sock);
    if (mr) ibv_dereg_mr(mr);
    free(buf);
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
    struct ibv_mr *mr = NULL;
    struct ibv_ah *ah = NULL;
    void *buf = NULL;
    uint8_t port_num = 0;
    char dev_name[64];
    union ibv_gid local_gid;
    int sock = -1;
    int rc = 1;

    if (find_active_roce(&ctx, &port_num, dev_name, sizeof(dev_name), want_dev)) {
        fprintf(stderr, "No active RoCE port\n");
        return 1;
    }
    printf("[client] %s port %u\n", dev_name, port_num);

    if (ibv_query_gid(ctx, port_num, GID_IDX, &local_gid)) {
        fprintf(stderr, "ibv_query_gid idx=%d failed\n", GID_IDX);
        goto cleanup;
    }

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
    printf("[client] remote DCT#=%u rkey=0x%x addr=%lx key=0x%016lx\n",
           info.dct_num, info.rkey, info.addr, info.dct_key);

    pd = ibv_alloc_pd(ctx);
    scq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    rcq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    if (!pd || !scq || !rcq) goto cleanup;

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

    dci = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!dci) goto cleanup;

    struct ibv_port_attr pa;
    ibv_query_port(ctx, port_num, &pa);

    struct ibv_qp_attr qpa = {};
    qpa.qp_state = IBV_QPS_INIT;
    qpa.pkey_index = 0;
    qpa.port_num = port_num;
    if (ibv_modify_qp(dci, &qpa,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT))
        goto cleanup;

    memset(&qpa, 0, sizeof(qpa));
    qpa.qp_state = IBV_QPS_RTR;
    qpa.path_mtu = pa.active_mtu;
    qpa.ah_attr.is_global = 1;
    qpa.ah_attr.port_num = port_num;
    qpa.ah_attr.grh.dgid = local_gid;
    qpa.ah_attr.grh.sgid_index = GID_IDX;
    qpa.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(dci, &qpa, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV))
        goto cleanup;

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
                          IBV_QP_MAX_QP_RD_ATOMIC))
        goto cleanup;

    printf("[client] DCI#=%u gid_idx=%d\n", dci->qp_num, GID_IDX);

    buf = calloc(1, BUF_SIZE);
    mr = ibv_reg_mr(pd, buf, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!buf || !mr) goto cleanup;
    snprintf(buf, BUF_SIZE, "Hello DC from pid=%d on %s", getpid(), dev_name);

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
    size_t len = strlen(buf) + 1;

    ibv_wr_start(qpx);
    qpx->wr_id = 1;
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qpx, info.rkey, info.addr);
    mlx5dv_wr_set_dc_addr(dv, ah, info.dct_num, info.dct_key);
    ibv_wr_set_sge(qpx, mr->lkey, (uint64_t)(uintptr_t)buf, len);
    if (ibv_wr_complete(qpx)) goto cleanup;
    printf("[client] RDMA Write posted, %zu bytes\n", len);

    struct ibv_wc wc;
    int got = 0;
    for (int i = 0; i < POLL_RETRIES; i++) {
        int n = ibv_poll_cq(scq, 1, &wc);
        if (n > 0) { got = 1; break; }
        if (n < 0) goto cleanup;
    }
    if (!got || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "CQ error: status=%d vendor=0x%x\n", wc.status,
                wc.vendor_err);
        goto cleanup;
    }
    printf("[client] CQE OK opcode=%d wr_id=%lu\n", wc.opcode, wc.wr_id);

    write(sock, "DONE", 4);
    printf("\n*** Client PASSED ***\n");
    rc = 0;

cleanup:
    if (sock >= 0) close(sock);
    if (ah) ibv_destroy_ah(ah);
    if (dci) ibv_destroy_qp(dci);
    if (mr) ibv_dereg_mr(mr);
    free(buf);
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
            "  loopback: %s\n"
            "  server:   %s -s [-p port] [-d dev]\n"
            "  client:   %s -c -a <ip> [-p port] [-d dev]\n",
            p, p, p);
}

int main(int argc, char **argv)
{
    int is_server = 0, is_client = 0, tcp_port = TCP_PORT;
    char *server_ip = NULL, *want_dev = NULL;
    int opt;

    while ((opt = getopt(argc, argv, "sca:p:d:h")) != -1) {
        switch (opt) {
        case 's': is_server = 1; break;
        case 'c': is_client = 1; break;
        case 'a': server_ip = optarg; break;
        case 'p': tcp_port = atoi(optarg); break;
        case 'd': want_dev = optarg; break;
        default: usage(argv[0]); return 1;
        }
    }

    if (is_server && is_client) {
        usage(argv[0]);
        return 1;
    }
    if (is_server)
        return run_server(tcp_port, want_dev);
    if (is_client) {
        if (!server_ip) {
            usage(argv[0]);
            return 1;
        }
        return run_client(server_ip, tcp_port, want_dev);
    }

    /* no -s/-c: run original loopback via exec sibling binary if present */
    fprintf(stderr, "No mode: use -s (server) or -c (client), or run ./dct_test for loopback\n");
    return 1;
}
