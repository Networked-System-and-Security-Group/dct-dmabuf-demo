# DCT DMA-BUF Demo

Minimal Mellanox/NVIDIA mlx5 DC QP demos for validating:

- DCT/DCI creation and DC RDMA Write
- split sender/receiver DC Write across two hosts
- GPU memory registration through CUDA DMA-BUF
- the perftest-style GPU DMA-BUF address layout (`base + 64 KiB`)
- basic data consistency under repeated sends

The code was tested on hosts `192.168.5.112` and `192.168.5.113`, using the
active RoCE device `mlx5_1` and RoCEv2 IPv4 GID index `3`.

## Files

- `dct_test.c`: single-process host-memory DCT/DCI loopback smoke test.
- `dct_test_remote.c`: split host-memory DCT server / DCI client test.
- `dct_test_gpu.c`: split DCT/DCI test using CUDA DMA-BUF registered GPU memory.

## Requirements

- Mellanox/NVIDIA mlx5 NIC with DC transport support
- `libibverbs` and `mlx5dv` headers/libraries
- CUDA driver/runtime headers and libraries for `dct_test_gpu`
- RoCEv2 GID index selected correctly; on the tested hosts this is `3`

Useful environment checks:

```bash
ibv_devinfo | grep -E 'hca_id|state:|link_layer'
show_gids | grep mlx5_1
nvidia-smi -L
```

## Build

```bash
make
```

Equivalent manual commands:

```bash
gcc -O2 -o dct_test dct_test.c -libverbs -lmlx5
gcc -O2 -o dct_test_remote dct_test_remote.c -libverbs -lmlx5
gcc -O2 -o dct_test_gpu dct_test_gpu.c -libverbs -lmlx5 \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -lcuda -lcudart -Wl,-rpath,/usr/local/cuda/lib64
```

## Host-Memory Tests

Loopback on one host:

```bash
./dct_test
```

Split sender/receiver:

```bash
# receiver, e.g. 113
./dct_test_remote -s -p 19998 -d mlx5_1

# sender, e.g. 112
./dct_test_remote -c -a 10.99.3.3 -p 19998 -d mlx5_1
```

Expected result: client sees `CQE OK`, and server prints the received buffer.

## GPU DMA-BUF Test

Copy the binary to the receiver host:

```bash
scp dct_test_gpu 192.168.5.113:/tmp/dct_test_gpu
```

Run a GPU-to-GPU DC RDMA Write:

```bash
# receiver, e.g. 113
/tmp/dct_test_gpu -s -p 19997 -d mlx5_1 -g 0

# sender, e.g. 112
./dct_test_gpu -c -a 10.99.3.3 -p 19997 -d mlx5_1 -g 0
```

Expected result:

- client: `*** Client PASSED (GPU dmabuf) ***`
- server: prints `GPU buffer: "..."`

## Stress / Consistency Tests

The GPU test supports repeated transfers and per-iteration verification:

```bash
# receiver
/tmp/dct_test_gpu -s -p 19997 -d mlx5_1 -g 0 -n 1000 -z 64

# sender, with cudaDeviceSynchronize before every post
./dct_test_gpu -c -a 10.99.3.3 -p 19997 -d mlx5_1 -g 0 -n 1000 -z 64
```

Disable the sender-side CUDA sync:

```bash
./dct_test_gpu -c -a 10.99.3.3 -p 19997 -d mlx5_1 -g 0 -n 1000 -z 64 -N
```

Tested results on 112 -> 113:

| Payload | Iterations | Sender sync | Result |
| --- | ---: | --- | --- |
| 64 B | 1000 | on | passed, `verify_fail=0` |
| 64 B | 1000 | off | failed, hundreds of verify errors |
| 4 KiB | 1000 | on | passed, `verify_fail=0` |
| 4 KiB | 1000 | off | passed in the tested run |
| 64 KiB | 1000 | on | passed, `verify_fail=0` |
| 64 KiB | 1000 | off | passed in the tested run |

The 64 B no-sync failure is intentional and useful: send CQE success does not
prove the remote GPU sees the intended data if the sender-side GPU writes have
not been made visible to the NIC.

## DMA-BUF Offset

`dct_test_gpu.c` intentionally follows the perftest CUDA DMA-BUF layout:

```c
#define GPU_REG_SIZE 131072
#define GPU_DATA_OFF 65536
```

The program registers a 128 KiB CUDA DMA-BUF region using the CUDA allocation
base as IOVA, but it performs RDMA from/to `base + 64 KiB`.

This matters. Using the raw CUDA allocation base as the RDMA address produced
successful send CQEs but empty or stale remote data in our tests. With the
`base + 64 KiB` data offset, cross-host GPU-to-GPU, GPU-to-host, and
host-to-GPU transfers all succeeded.

The offset is part of the buffer layout used for both the local SGE and the
remote RDMA address, so it does not change data consistency by itself. It only
selects a valid data subrange inside the registered DMA-BUF mapping. Consistency
still depends on proper sender-side GPU synchronization/flush before NIC reads
and receiver-side visibility/invalidation before GPU consumers read data.

## Known Limitations

- Single-process GPU DMA-BUF loopback (`-l`) is not used as the correctness
  criterion. Cross-host split sender/receiver is the target scenario.
- `cudaDeviceSynchronize()` is used as a simple demonstration barrier. A
  production implementation should use the correct CUDA/GDR memory visibility
  primitives for the actual data path.
- The code fixes GID index to `3` because that is the RoCEv2 IPv4 GID on the
  tested machines. Adjust it if your environment differs.
