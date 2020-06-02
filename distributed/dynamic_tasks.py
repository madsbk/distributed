import pandas as pd
from dask import persist
from dask.base import tokenize
import dask.dataframe as dd
from distributed import Client, LocalCluster, get_worker, get_client
from distributed.worker import dumps_task
import pickle
import dask
from dask.datasets import timeseries
from dask.dataframe.shuffle import shuffle, partitioning_index, shuffle_group, _concat
from dask.dataframe import _Frame
import time
from operator import getitem


def dynshuffle_kernel(df, npartitions, kernel_token, rearguard_token, col):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    myid = int(myself.split(",")[1][:-1])
    parts = [str((kernel_token[1:], i)) for i in range(npartitions)]
    rearguard = str((rearguard_token[1:], myid))

    print(
        f"[{worker.address}] kernel() - myid: {myid}, key: {repr(myself)}, parts: {parts}, rearguard: {rearguard}"
    )

    groups = shuffle_group(df, col, 0, len(parts), len(parts), ignore_index=False)
    assert len(groups) == len(parts)

    new_tasks = []
    for i, part in enumerate(parts):
        new_tasks.append(
            {
                "key": f"shuffle_getitem_{myself}_{part}",
                "deps": [part],
                "task": dumps_task((getitem, part, i)),
                "priority": 0,
            }
        )
    getitem_keys = [t["key"] for t in new_tasks]

    new_tasks.append(
        {
            "key": f"shuffle_join_{myself}",
            "deps": [myself] + getitem_keys,
            "task": dumps_task((_concat, getitem_keys)),
        }
    )

    client.sync(
        worker.scheduler.extend_current_task,
        cur_key=myself,
        new_tasks=new_tasks,
        rearguard_key=rearguard,
        rearguard_input=f"shuffle_join_{myself}",
    )

    return groups

def dummy_func():
    pass

def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    kernel_token = "dynshuffle-%s" % tokenize(df, column, max_branch, npartitions, ignore_index)
    rearguard_token = "rearguard_%s" % kernel_token
    res = df.map_partitions(
        dynshuffle_kernel,
        npartitions=df.npartitions,
        kernel_token="_%s" % kernel_token,
        rearguard_token="_%s" % rearguard_token,
        col="_partitions",
        meta=df._meta,
        full_token=kernel_token,
    )
    res = res.map_partitions(dummy_func, meta=res._meta, full_token=rearguard_token)
    return res


