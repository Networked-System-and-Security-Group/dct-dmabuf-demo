/*
 * dct_test.c — Minimal DC (Dynamically Connected) transport test.
 *
 * Validates DCT + DCI creation and loopback RDMA Write using the patterns
 * found in perftest (perftest_resources.c).
 *
 * Key patterns from perftest:
 *   - DCT: srq attached, dc_type=DCT, random dct_access_key, stays at RTR
 *   - DCI: DISABLE_SCATTER_TO_CQE, dc_type=DCI, SEND_OPS_FLAGS, goes to RTS
 *   - DCI RESET→INIT skips access_flags
 *   - Per-WR target: mlx5dv_wr_set_dc_addr(dv_qp, ah, remote_dctn, dc_key)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#define CQ_DEPTH    128
#define BUF_SIZE    4096
#define POLL_RETRIES 100000

/* ================================================================
 * helpers
 * ================================================================ */
static void dump_hex(const char *label, const void *buf, int len) {
    printf("%s: ", label);
    const unsigned char *p = (const unsigned char *)buf;
    for (int i = 0; i < len && i < 16; i++)
        printf("%02x ", p[i]);
    if (len > 16) printf("...");
    printf("\n");
}

#define CHECK(call, msg) do { \
    if (!(call)) { fprintf(stderr, "FAIL: %s: %s\n", msg, strerror(errno)); goto cleanup; } \
} while(0)

#define CHECK_RET(call, msg) do { \
    if ((call) != 0) { fprintf(stderr, "FAIL: %s: %s\n", msg, strerror(errno)); goto cleanup; } \
} while(0)

/* ================================================================ */
int main(void) {
    int rc = EXIT_FAILURE;

    /* --- pointers to clean up --- */
    struct ibv_device      **dev_list = NULL;
    struct ibv_context      *ctx     = NULL;
    struct ibv_pd           *pd      = NULL;
    struct ibv_cq           *scq     = NULL; /* send CQ (for DCI) */
    struct ibv_cq           *rcq     = NULL; /* recv CQ (for DCT) */
    struct ibv_srq          *srq     = NULL;
    struct ibv_qp           *dct     = NULL;
    struct ibv_qp           *dci     = NULL;
    struct ibv_mr           *dct_mr  = NULL;
    struct ibv_mr           *dci_mr  = NULL;
    struct ibv_ah           *ah      = NULL;
    void                    *dct_buf = NULL;
    void                    *dci_buf = NULL;

    uint32_t dct_num = 0, dci_num = 0;
    uint8_t  port_num = 0;
    char     dev_name[64] = {0};
    union ibv_gid local_gid;

    srand((unsigned)(time(NULL) ^ getpid()));
    uint64_t dct_key = ((uint64_t)rand() << 32) | (uint64_t)rand();

    /* ============================================================
     * 1. Find first active RoCE port
     * ============================================================ */
    {
        int ndev;
        dev_list = ibv_get_device_list(&ndev);
        if (!dev_list || ndev == 0) {
            fprintf(stderr, "No IB devices\n");
            goto cleanup;
        }
        printf("Found %d device(s)\n", ndev);

        int found = 0;
        for (int i = 0; i < ndev && !found; i++) {
            ctx = ibv_open_device(dev_list[i]);
            if (!ctx) continue;

            struct ibv_device_attr da;
            CHECK_RET(ibv_query_device(ctx, &da), "ibv_query_device");

            for (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
                struct ibv_port_attr pa;
                if (ibv_query_port(ctx, p, &pa)) continue;
                if (pa.state != IBV_PORT_ACTIVE) continue;
                if (pa.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

                port_num = p;
                snprintf(dev_name, sizeof(dev_name), "%s",
                         ibv_get_device_name(dev_list[i]));
                printf("Using %s port %u (RoCE ACTIVE, mtu=%d)\n",
                       dev_name, port_num, pa.active_mtu);
                found = 1;
                break;
            }
            if (!found) { ibv_close_device(ctx); ctx = NULL; }
        }
        if (!found) { fprintf(stderr, "No active RoCE port\n"); goto cleanup; }
    }

    /* ============================================================
     * 2. PD
     * ============================================================ */
    pd = ibv_alloc_pd(ctx);
    CHECK(pd, "ibv_alloc_pd");

    /* ============================================================
     * 3. CQs — separate send & recv (perftest pattern)
     * ============================================================ */
    scq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    CHECK(scq, "ibv_create_cq send");
    rcq = ibv_create_cq(ctx, CQ_DEPTH, NULL, NULL, 0);
    CHECK(rcq, "ibv_create_cq recv");

    /* ============================================================
     * 4. SRQ — mandatory for DC
     *    perftest: ibv_create_srq_ex, IBV_SRQT_BASIC, max_wr=rx_depth
     * ============================================================ */
    {
        struct ibv_srq_init_attr_ex sa;
        memset(&sa, 0, sizeof(sa));
        sa.comp_mask          = IBV_SRQ_INIT_ATTR_TYPE | IBV_SRQ_INIT_ATTR_PD;
        sa.attr.max_wr        = 64;
        sa.attr.max_sge       = 1;
        sa.pd                 = pd;
        sa.srq_type           = IBV_SRQT_BASIC;

        srq = ibv_create_srq_ex(ctx, &sa);
        CHECK(srq, "ibv_create_srq_ex");
        printf("SRQ created\n");
    }

    /* ============================================================
     * 5. Query local GID (RoCEv2)
     * ============================================================ */
    {
        /* try GID index 3 first (RoCEv2), fallback to 0 */
        int gid_idx = 3;
        if (ibv_query_gid(ctx, port_num, gid_idx, &local_gid)) {
            gid_idx = 0;
            CHECK_RET(ibv_query_gid(ctx, port_num, 0, &local_gid),
                      "ibv_query_gid");
        }
        printf("Local GID index=%d ready\n", gid_idx);
    }

    /* ============================================================
     * 6. Create DCT — perftest pattern
     * ============================================================ */
    {
        struct ibv_qp_init_attr_ex attr_ex;
        memset(&attr_ex, 0, sizeof(attr_ex));
        attr_ex.qp_type          = IBV_QPT_DRIVER;
        attr_ex.send_cq          = scq;
        attr_ex.recv_cq          = rcq;
        attr_ex.srq              = srq;          /* DCT 挂 SRQ */
        attr_ex.cap.max_send_wr  = 1;
        attr_ex.cap.max_recv_wr  = 64;
        attr_ex.cap.max_send_sge = 1;
        attr_ex.cap.max_recv_sge = 1;
        attr_ex.pd               = pd;
        attr_ex.comp_mask        = IBV_QP_INIT_ATTR_PD;
        /* DCT: no send ops */

        struct mlx5dv_qp_init_attr attr_dv;
        memset(&attr_dv, 0, sizeof(attr_dv));
        attr_dv.comp_mask                  = MLX5DV_QP_INIT_ATTR_MASK_DC;
        attr_dv.dc_init_attr.dc_type       = MLX5DV_DCTYPE_DCT;
        attr_dv.dc_init_attr.dct_access_key = dct_key;

        dct = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
        CHECK(dct, "mlx5dv_create_qp(DCT)");
        printf("DCT created (qp_num placeholder=%u, after RTR real num assigned)\n",
               dct->qp_num);
    }

    /* DCT: RESET → INIT (perftest: DCT sets access_flags) */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = port_num;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ  |
                               IBV_ACCESS_REMOTE_ATOMIC;

        CHECK_RET(ibv_modify_qp(dct, &attr,
                                IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
                  "DCT RESET→INIT");
        printf("DCT: RESET → INIT OK\n");
    }

    /* ============================================================
     * 7. Create DCI — perftest pattern
     *
     * Critical: DC skips recv setup on DCI → max_recv_wr=0, max_recv_sge=0.
     *           send_ops_flags must match the actual verb (WRITE).
     *           MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE is set.
     * ============================================================ */
    {
        struct ibv_qp_init_attr_ex attr_ex;
        memset(&attr_ex, 0, sizeof(attr_ex));
        attr_ex.qp_type          = IBV_QPT_DRIVER;
        attr_ex.send_cq          = scq;
        attr_ex.recv_cq          = rcq;
        attr_ex.srq              = NULL;          /* DCI: no SRQ */
        attr_ex.cap.max_send_wr  = 128;
        attr_ex.cap.max_recv_wr  = 0;             /* DC skips recv for DCI */
        attr_ex.cap.max_send_sge = 2;             /* MAX_SEND_SGE */
        attr_ex.cap.max_recv_sge = 0;             /* DC skips recv for DCI */
        attr_ex.cap.max_inline_data = 0;
        attr_ex.pd               = pd;
        attr_ex.comp_mask        = IBV_QP_INIT_ATTR_PD |
                                   IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        attr_ex.send_ops_flags   = IBV_QP_EX_WITH_RDMA_WRITE;

        struct mlx5dv_qp_init_attr attr_dv;
        memset(&attr_dv, 0, sizeof(attr_dv));
        attr_dv.comp_mask   = MLX5DV_QP_INIT_ATTR_MASK_DC |
                              MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        attr_dv.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
        attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
        attr_dv.dc_init_attr.dci_streams.log_num_concurent = 0;
        attr_dv.dc_init_attr.dci_streams.log_num_errored   = 0;

        dci = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
        CHECK(dci, "mlx5dv_create_qp(DCI)");
        dci_num = dci->qp_num;
        printf("DCI created, qp_num=%u\n", dci_num);
    }

    /* DCI: RESET → INIT (perftest: DCI skips access_flags!) */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state   = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num   = port_num;
        /* NO access_flags for DCI */

        CHECK_RET(ibv_modify_qp(dci, &attr,
                                IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                IBV_QP_PORT),
                  "DCI RESET→INIT");
        printf("DCI: RESET → INIT OK\n");
    }

    /* ============================================================
     * 8. Query port for MTU
     * ============================================================ */
    struct ibv_port_attr pa;
    CHECK_RET(ibv_query_port(ctx, port_num, &pa), "ibv_query_port");

    /* ============================================================
     * 9. DCT INIT → RTR
     *    perftest: path_mtu, AV (is_global, grh), min_rnr_timer
     * ============================================================ */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state       = IBV_QPS_RTR;
        attr.path_mtu       = pa.active_mtu;
        attr.min_rnr_timer  = 12;

        attr.ah_attr.is_global          = 1;
        attr.ah_attr.port_num           = port_num;
        attr.ah_attr.grh.dgid           = local_gid;
        attr.ah_attr.grh.sgid_index     = 3;
        attr.ah_attr.grh.hop_limit      = 64;
        attr.ah_attr.grh.traffic_class  = 0;

        CHECK_RET(ibv_modify_qp(dct, &attr,
                                IBV_QP_STATE | IBV_QP_MIN_RNR_TIMER |
                                IBV_QP_PATH_MTU | IBV_QP_AV),
                  "DCT INIT→RTR");
        dct_num = dct->qp_num;   /* real DCT number assigned after RTR */
        printf("DCT: INIT → RTR OK  DCT#=%u  key=0x%016lx\n",
               dct_num, dct_key);
    }

    /* ============================================================
     * 10. DCI INIT → RTR
     *     perftest: AV + MTU same as DCT, but NO min_rnr_timer for DCI
     * ============================================================ */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state  = IBV_QPS_RTR;
        attr.path_mtu  = pa.active_mtu;

        attr.ah_attr.is_global          = 1;
        attr.ah_attr.port_num           = port_num;
        attr.ah_attr.grh.dgid           = local_gid;
        attr.ah_attr.grh.sgid_index     = 3;
        attr.ah_attr.grh.hop_limit      = 64;
        attr.ah_attr.grh.traffic_class  = 0;

        CHECK_RET(ibv_modify_qp(dci, &attr,
                                IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV),
                  "DCI INIT→RTR");
        printf("DCI: INIT → RTR OK\n");
    }

    /* ============================================================
     * 11. DCI RTR → RTS (DCT stays at RTR, per perftest)
     *     perftest: sq_psn (applies to all non-RawEth),
     *               timeout, retry_cnt, rnr_retry, max_rd_atomic
     * ============================================================ */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state      = IBV_QPS_RTS;
        attr.sq_psn        = 0;  /* perftest: my_dest->psn; DC doesn't use it */
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7;
        attr.max_rd_atomic = 1;

        CHECK_RET(ibv_modify_qp(dci, &attr,
                                IBV_QP_STATE | IBV_QP_SQ_PSN |
                                IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC),
                  "DCI RTR→RTS");
        printf("DCI: RTR → RTS OK\n");
    }

    /* ============================================================
     * 12. Register MRs
     * ============================================================ */
    dct_buf = calloc(1, BUF_SIZE);
    CHECK(dct_buf, "calloc dct_buf");
    dct_mr = ibv_reg_mr(pd, dct_buf, BUF_SIZE,
                        IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ);
    CHECK(dct_mr, "ibv_reg_mr dct");

    dci_buf = calloc(1, BUF_SIZE);
    CHECK(dci_buf, "calloc dci_buf");
    memset(dci_buf, 0xAB, 64);  /* write pattern */
    dci_mr = ibv_reg_mr(pd, dci_buf, BUF_SIZE,
                        IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_WRITE);
    CHECK(dci_mr, "ibv_reg_mr dci");

    printf("MRs: dct=%p(lkey=0x%x,rkey=0x%x) dci=%p(lkey=0x%x,rkey=0x%x)\n",
           dct_buf, dct_mr->lkey, dct_mr->rkey,
           dci_buf, dci_mr->lkey, dci_mr->rkey);

    /* ============================================================
     * 13. Create AH for RoCE loopback
     * ============================================================ */
    {
        struct ibv_ah_attr ah_attr;
        memset(&ah_attr, 0, sizeof(ah_attr));
        ah_attr.is_global          = 1;
        ah_attr.port_num           = port_num;
        ah_attr.grh.dgid           = local_gid;
        ah_attr.grh.sgid_index     = 3;
        ah_attr.grh.hop_limit      = 64;
        ah_attr.grh.traffic_class  = 0;

        ah = ibv_create_ah(pd, &ah_attr);
        CHECK(ah, "ibv_create_ah");
        printf("AH created (RoCE loopback)\n");
    }

    /* ============================================================
     * 14. Post RDMA Write: DCI → DCT (loopback)
     *     perftest: wr_start → wr_flags=SIGNALED → wr_rdma_write →
     *               mlx5dv_wr_set_dc_addr → ibv_wr_set_sge → wr_complete
     * ============================================================ */
    {
        struct ibv_qp_ex      *qpx  = ibv_qp_to_qp_ex(dci);
        struct mlx5dv_qp_ex   *dv_qp = mlx5dv_qp_ex_from_ibv_qp_ex(qpx);

        ibv_wr_start(qpx);
        qpx->wr_id     = 1;
        qpx->wr_flags  = IBV_SEND_SIGNALED;  /* need CQE */

        ibv_wr_rdma_write(qpx, dct_mr->rkey, (uint64_t)(uintptr_t)dct_buf);

        /* perftest: set DC destination BEFORE sge (lines 518-535 before 562-586) */
        mlx5dv_wr_set_dc_addr(dv_qp, ah, dct_num, dct_key);

        ibv_wr_set_sge(qpx, dci_mr->lkey, (uint64_t)(uintptr_t)dci_buf, 64);

        CHECK_RET(ibv_wr_complete(qpx), "ibv_wr_complete");
        printf("RDMA Write posted: DCI#%u → DCT#%u, 64 bytes\n",
               dci_num, dct_num);
    }

    /* ============================================================
     * 15. Poll CQ (DCI send completion)
     *     RDMA Write is one-sided — only the initiator gets a CQE
     * ============================================================ */
    {
        struct ibv_wc wc;
        int got = 0;
        for (int i = 0; i < POLL_RETRIES; i++) {
            int n = ibv_poll_cq(scq, 1, &wc);
            if (n < 0) {
                fprintf(stderr, "FAIL: ibv_poll_cq: %s\n", strerror(errno));
                goto cleanup;
            }
            if (n > 0) { got = 1; break; }
        }
        CHECK(got, "CQ poll (timeout)");
        CHECK(wc.status == IBV_WC_SUCCESS, "completion status");

        printf("CQE: status=SUCCESS opcode=%d wr_id=%lu byte_len=%u src_qp=%u\n",
               wc.opcode, wc.wr_id, wc.byte_len, wc.src_qp);
    }

    /* ============================================================
     * 16. Verify data
     * ============================================================ */
    if (memcmp(dct_buf, dci_buf, 64) == 0) {
        printf("Data verified OK (64 bytes 0xAB written)\n");
    } else {
        printf("Data MISMATCH!\n");
        dump_hex("EXPECTED", dci_buf, 16);
        dump_hex("GOT     ", dct_buf, 16);
        goto cleanup;
    }

    printf("\n*** Test PASSED ***\n");
    rc = EXIT_SUCCESS;

cleanup:
    /* perftest destroy_ctx order: AH → QP → SRQ → CQ → MR → PD → close_device */
    if (ah)      ibv_destroy_ah(ah);
    if (dci)     ibv_destroy_qp(dci);
    if (dct)     ibv_destroy_qp(dct);
    if (srq)     ibv_destroy_srq(srq);
    if (rcq)     ibv_destroy_cq(rcq);
    if (scq)     ibv_destroy_cq(scq);
    if (dci_mr)  ibv_dereg_mr(dci_mr);
    if (dct_mr)  ibv_dereg_mr(dct_mr);
    free(dci_buf);
    free(dct_buf);
    if (pd)      ibv_dealloc_pd(pd);
    if (ctx)     ibv_close_device(ctx);
    if (dev_list) ibv_free_device_list(dev_list);

    return rc;
}
