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
from dask.dataframe.core import new_dd_object
import time
from operator import getitem
from dask.highlevelgraph import HighLevelGraph


def _dummy_func():
    pass


def _apply_func(func, df, func_parts_encoded, rearguard_encoded, kwargs):
    parts = [k[1:] for k in func_parts_encoded]
    rearguard = rearguard_encoded[1:]
    return func(df, parts, rearguard, **kwargs)


def dd_dynamic_tasks_map(func, ddf, meta, name, **kwargs):
    n = ddf.npartitions
    token = tokenize(ddf, func, meta, name)
    name = "%s-%s" % (name, token)
    df_parts = [(ddf._name, i) for i in range(n)]
    func_parts = [(name, i) for i in range(n)]
    func_parts_encoded = ["_" + str(k) for k in func_parts]
    rearguard_parts = [("rearguard_" + name, i) for i in range(n)]

    layer = {}
    for func_part, rearguard_part, df_part in zip(
        func_parts, rearguard_parts, df_parts
    ):
        layer[func_part] = (
            _apply_func,
            func,
            df_part,
            func_parts_encoded,
            "_" + str(rearguard_part),
            kwargs,
        )

    ddf2 = new_dd_object(
        HighLevelGraph.from_collections(name, layer, dependencies=[ddf]),
        name,
        ddf._meta,
        ddf.divisions,
    )

    layer = {}
    for func_part, rearguard_part, df_part in zip(
        func_parts, rearguard_parts, df_parts
    ):
        layer[rearguard_part] = (_dummy_func, func_part)

    ddf3 = new_dd_object(
        HighLevelGraph.from_collections(
            "rearguard_" + name, layer, dependencies=[ddf2]
        ),
        "rearguard_" + name,
        ddf2._meta,
        ddf.divisions,
    )
    print("HEJ")
    return ddf3


def dynshuffle_kernel(df, parts, rearguard, col, ignore_index):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    assert myself in parts

    print(
        f"[{worker.address}] _apply_func() - key: {repr(myself)}, df: {type(df)}, parts: {parts}, rearguard: {rearguard}, col: {repr(col)}"
    )

    groups = shuffle_group(
        df, col, 0, len(parts), len(parts), ignore_index=ignore_index, nfinal=len(parts)
    )
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


def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    print(f"rearrange_by_column_dynamic_tasks() - column: {column}, \nddf: {df.compute()}")
    return dd_dynamic_tasks_map(
        dynshuffle_kernel, df, df._meta, "dynshuffle", col=column, ignore_index=ignore_index
    )





def kernel(df, peers, rearguard, col):
    worker = get_worker()
    client = get_client()
    peers = [p[1:] for p in peers]
    rearguard = rearguard[1:]
    myself = get_worker().get_current_task()

    print(
        f"[{worker.address}] kernel() - key: {myself}, peers: {peers}, rearguard: {rearguard}"
    )

    groups = shuffle_group(df, col, 0, len(peers), len(peers), ignore_index=False, nfinal=len(peers))
    assert len(groups) == len(peers)

    new_tasks = []
    for i, peer in enumerate(peers):
        new_tasks.append(
            {
                "key": f"shuffle_getitem_{myself}_{peer}",
                "deps": [peer],
                "task": dumps_task((getitem, peer, i)),
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


def noop(df):
    return df




def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    print(f"rearrange_by_column_dynamic_tasks() - column: {column}, \nddf: {df.compute()}")

    token = tokenize(df, column, max_branch, npartitions, ignore_index)
    delayed_df = df.to_delayed()
    delayed_kernel = dask.delayed(kernel)
    rearguard = dask.delayed(noop)
    kernel_names = [f"shuffle_{i}_{token}" for i in range(len(delayed_df))]
    kernel_names_encoded = ["_shuffle_%d" % i for i in range(len(delayed_df))]
    rearguard_names_encoded = ["%s_rearguard" % k for k in kernel_names_encoded]

    res = [
        rearguard(
            delayed_kernel(
                d,
                kernel_names_encoded,
                f"{kernel_names_encoded[i]}_rearguard",
                column,
                dask_key_name=kernel_names[i],
            ),
            dask_key_name=f"{kernel_names[i]}_rearguard",
        )
        for i, d in enumerate(delayed_df)
    ]
    print(res)

    return dd.from_delayed(res, meta=df._meta)
