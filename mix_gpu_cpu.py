import argparse
import asyncio
import os
import time
from typing import Iterable, List, Mapping, Optional

import pandas as pd
from dask_cuda.utils import (
    CPUAffinity,
    RMMSetup,
    cuda_visible_devices,
    get_cpu_affinity,
    nvml_device_index,
)

import dask
import dask.array.random
import dask.dataframe as dd
from dask.distributed import Client, Nanny, Scheduler, wait
from dask.utils import parse_bytes

os.environ["RAPIDS_NO_INITIALIZE"] = "True"
import cudf


def local_mixtyped_object(
    obj,
    partition2type: Iterable[Optional[type]],
    type_converters: Mapping[type, callable],
    partition_info: Mapping,
):
    partition_number = partition_info["number"]
    assert 0 <= partition_number and partition_number < len(partition2type)
    target_type = partition2type[partition_number]

    if target_type is None or isinstance(obj, target_type):
        return obj  # Already the correct type
    else:
        ret = type_converters[target_type](obj)
        assert isinstance(ret, target_type)
        return ret


def mixtyped_dataframe(
    df: dd.Series,
    type_fractions: Mapping[type, float],
    type_converters: Mapping[type, callable],
    meta=None,
):
    meta = df._meta if meta is None else meta
    npartitions = df.npartitions
    partition2type: List[type] = []
    for t, f in type_fractions.items():
        n = int(npartitions * f)
        partition2type.extend([t] * n)
    rest = npartitions - len(partition2type)
    if rest:
        partition2type.extend([None] * rest)
    ret = df.map_partitions(
        local_mixtyped_object, partition2type, type_converters, meta=meta
    )
    return ret


def work(df):
    print(
        f"work() - df: {type(df)}, CUDA_VISIBLE_DEVICES: ",
        os.getenv("CUDA_VISIBLE_DEVICES"),
    )
    return df.sort_values()


def cudf_to_pandas(df: cudf.Series):
    return df.to_pandas()


def pandas_to_cudf(df: pd.Series):
    return cudf.Series(df)


type_converters = {pd.Series: cudf_to_pandas, cudf.Series: pandas_to_cudf}

from dask.dataframe.shuffle import shuffle

async def test(args):
    size = args.nbytes // 8
    df = dd.from_dask_array(
        dask.array.random.random_sample(size, chunks=size // args.npartitions)
    )
    type_fractions = {
        pd.Series: args.cpu_fraction,
        cudf.Series: args.gpu_fraction,
    }
    df = mixtyped_dataframe(df, type_fractions, type_converters).persist()
    await wait(df)
    t1 = time.time()
    print("*" * 100, "STARTING TEST", "*" * 100)

    #res = df.map_partitions(work, meta=df._meta)
    #res = df.repartition(npartitions=1)

    df = df.to_frame("data")
    res = shuffle(
        df, "data", shuffle="tasks", npartitions=10
    )

    res = res.persist()
    await wait(res)
    t2 = time.time()
    print("test: ", t2 - t1)


async def main(s: Scheduler, c: Client, args):
    cpu_workers = []
    gpu_workers = []

    for _ in range(args.cpus):
        cpu_workers.append(
            Nanny(
                s.address,
                nthreads=1,
                memory_limit="20GB",
                resources={"CPU": 1, "GPU": 0},
            )
        )
        await cpu_workers[-1].start()

    for i in range(args.gpus):
        visible_devices = cuda_visible_devices(i)
        gpu_workers.append(
            Nanny(
                s.address,
                nthreads=1,
                memory_limit="20GB",
                resources={"CPU": 0, "GPU": 1},
                env={"CUDA_VISIBLE_DEVICES": visible_devices},
                preload=["dask_cuda.initialize"],
                preload_argv=["--create-cuda-context"],
                plugins={
                    CPUAffinity(
                        get_cpu_affinity(nvml_device_index(0, visible_devices))
                    ),
                    RMMSetup(10e9, False, False, None),
                },
            )
        )
        await gpu_workers[-1].start()

    print("*" * 100, "  INIT TEST  ", "*" * 100)
    await test(args)

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
    parser.add_argument("gpus", metavar="GPUS", type=int, help="Number of GPU workers")
    parser.add_argument(
        "--size",
        metavar="BYTES",
        type=str,
        default="1MB",
        help='Size in bytes (default: "1MB")',
    )
    parser.add_argument(
        "--npartitions",
        metavar="N",
        type=int,
        default=-1,
        help="Number of partitions (default: CPUS+GPUS)",
    )
    parser.add_argument(
        "--cpu-fraction",
        metavar="FRACTION",
        type=float,
        default=0.2,
        help="Fraction of CPU partitions (default: 0.2)",
    )
    parser.add_argument(
        "--gpu-fraction",
        metavar="FRACTION",
        type=float,
        default=0.8,
        help="Fraction of GPU partitions (default: 0.8)",
    )
    args = parser.parse_args()
    args.nworkers = args.cpus + args.gpus
    if args.npartitions == -1:
        args.npartitions = args.nworkers
    args.nbytes = parse_bytes(args.size)
    asyncio.get_event_loop().run_until_complete(launcher(args))
