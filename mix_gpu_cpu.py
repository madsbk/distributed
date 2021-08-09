import asyncio
import threading
from typing import List
import dask
from dask.delayed import delayed
from dask.distributed import Client, Scheduler, Worker, Nanny, wait
from dask.utils import stringify
from distributed.threadpoolexecutor import ThreadPoolExecutor
import dask.dataframe as dd
import argparse
import numpy as np
import pandas as pd
import cupy as cp
import time
import os
from dask_cuda.utils import (
    CPUAffinity,
    RMMSetup,
    _ucx_111,
    cuda_visible_devices,
    get_cpu_affinity,
    get_ucx_config,
    get_ucx_net_devices,
    nvml_device_index,
    parse_cuda_visible_device,
    parse_device_memory_limit,
)


os.environ["RAPIDS_NO_INITIALIZE"] = "True"
import cudf


def create_mixed_dataframe(args) -> dd:
    def pandas_df():
        return pd.DataFrame({"data": np.random.random_sample(args.cpu_partition_size)})

    def cudf_df():
        return cudf.DataFrame(
            {"data": cp.random.random_sample(args.gpu_partition_size)}
        )

    ret = []
    for _ in range(args.cpu_chunks):
        ret.append(delayed(pandas_df)())
    for _ in range(args.gpu_chunks):
        ret.append(delayed(cudf_df)())

    if args.gpus:
        meta = cudf.DataFrame({"data": cp.random.random_sample(1)})
    else:
        meta = pd.DataFrame({"data": np.random.random_sample(1)})

    return dd.from_delayed(ret, meta=meta, verify_meta=False)


def work(df):
    print(
        f"work() - df: {type(df)}, CUDA_VISIBLE_DEVICES: ",
        os.getenv("CUDA_VISIBLE_DEVICES"),
    )
    if isinstance(df, pd.DataFrame):
        # time.sleep(20)
        ret = df.sort_values("data")
    else:
        ret = df.sort_values("data")
        # time.sleep(1)
    return ret


async def test1(args):
    # rmm.reinitialize(initial_pool_size=5e9)
    df1: dd = create_mixed_dataframe(args).persist()
    await wait(df1)
    print("df1 created")
    t1 = time.time()

    res = df1.map_partitions(work, meta=df1._meta)

    res = res.persist()
    await wait(res)
    t2 = time.time()
    print("test1: ", t2 - t1)


async def main(s: Scheduler, c: Client, args):
    cpu_workers = []
    gpu_workers = []

    for _ in range(args.cpus):
        cpu_workers.append(Nanny(s.address, nthreads=1, resources={"CPU": 1},))
        await cpu_workers[-1].start()

    for i in range(args.gpus):
        visible_devices = cuda_visible_devices(i)
        gpu_workers.append(
            Nanny(
                s.address,
                nthreads=1,
                resources={"GPU": 1},
                env={"CUDA_VISIBLE_DEVICES": visible_devices},
                preload=["dask_cuda.initialize"],
                preload_argv=["--create-cuda-context"],
                plugins={
                    CPUAffinity(
                        get_cpu_affinity(nvml_device_index(0, visible_devices))
                    ),
                    RMMSetup(5e9, False, False, None),
                },
            )
        )
        await gpu_workers[-1].start()

    print("*" * 100, "STARTING TEST", "*" * 100)
    await test1(args)

    print("*" * 100, "SHUTDOWN TEST", "*" * 100)
    for nanny in cpu_workers + gpu_workers:
        await nanny.kill()


async def launcher(args):
    async with Scheduler() as s:
        async with Client(s.address, asynchronous=True) as c:
            await main(s, c, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("cpus", metavar="CPUS", type=int, help="Number of CPU workers")
    parser.add_argument(
        "cpu_chunks", metavar="CPUS", type=int, help="Number of CPU workers"
    )
    parser.add_argument("gpus", metavar="GPUS", type=int, help="Number of GPU workers")
    parser.add_argument(
        "gpu_chunks", metavar="CPUS", type=int, help="Number of CPU workers"
    )
    parser.add_argument(
        "--cpu-partition-size",
        metavar="SIZE",
        type=int,
        default=10,
        help="Size of each CPU partition",
    )
    parser.add_argument(
        "--gpu-partition-size",
        metavar="SIZE",
        type=int,
        help="Size of each GPU partition",
    )
    args = parser.parse_args()
    args.nworkers = args.cpus + args.gpus
    asyncio.get_event_loop().run_until_complete(launcher(args))
