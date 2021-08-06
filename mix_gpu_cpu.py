import asyncio
import threading
from typing import List
import dask
from dask.delayed import delayed
from dask.distributed import Client, Scheduler, Worker, Nanny, wait
from dask.utils import stringify
from distributed.threadpoolexecutor import ThreadPoolExecutor
import dask.dataframe as dd
import dask_cudf
import argparse
import numpy as np
import pandas as pd
import cupy as cp
import cudf
import time


def get_thread_name(prefix):
    return prefix + threading.current_thread().name


def f(df):
    print(f"{threading.current_thread().name}:")
    print(type(df))

    time.sleep(1)
    return df


def work(df):
    if isinstance(df, pd.DataFrame):
        time.sleep(2)
    else:
        time.sleep(1)
    return df


def create_mixed_dataframe(args) -> dd:
    ret = []
    for _ in range(args.cpus):
        ret.append(
            delayed(pd.DataFrame)(
                {"data": np.random.random_sample(args.cpu_partition_size)}
            )
        )
    for _ in range(args.gpus):
        ret.append(
            delayed(cudf.DataFrame)(
                {"data": cp.random.random_sample(args.gpu_partition_size)}
            )
        )
    if args.gpus:
        meta = cudf.DataFrame({"data": cp.random.random_sample(1)})
    else:
        meta = pd.DataFrame({"data": np.random.random_sample(1)})
    return dd.from_delayed(ret, meta=meta, verify_meta=False)


async def test1(args):
    df1: dd = create_mixed_dataframe(args).persist()
    await wait(df1)

    res = df1.map_partitions(work, meta=df1._meta)

    res = res.persist()
    await wait(res)


async def main(s: Scheduler, c: Client, args):
    cpu_workers = []
    gpu_workers = []

    cpu_workers.append(
        Nanny(s.address, nthreads=args.cpus, resources={"CPU": args.cpus})
    )
    await cpu_workers[0].start()

    for i in range(args.gpus):
        gpu_workers.append(
            Nanny(
                s.address,
                nthreads=1,
                resources={"GPU": 1},
                env={"CUDA_VISIBLE_DEVICES": i},
            )
        )
        await gpu_workers[-1].start()

    await test1(args)

    for nanny in cpu_workers + gpu_workers:
        await nanny.kill()


async def launcher(args):
    async with Scheduler() as s:
        async with Client(s.address, asynchronous=True) as c:
            await main(s, c, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("cpus", metavar="CPUS", type=int, help="Number of CPU workers")
    parser.add_argument("gpus", metavar="GPUS", type=int, help="Number of GPU workers")
    parser.add_argument(
        "--npartitions", metavar="PARTITIONS", type=int, help="Number of partitions"
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
