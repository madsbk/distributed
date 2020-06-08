import pandas as pd
from dask import persist
from dask.base import tokenize
import dask.dataframe as dd
from distributed import Client, LocalCluster, get_worker, get_client
from distributed.worker import dumps_task
import pickle
import dask
from dask.datasets import timeseries
from dask.dataframe.shuffle import (
    shuffle,
    partitioning_index,
    shuffle_group,
    _concat,
    shuffle_group_2,
    shuffle_group_get,
)
from dask.dataframe import _Frame
from dask.dataframe.core import new_dd_object
import time
from operator import getitem
from dask.highlevelgraph import HighLevelGraph


def _dummy_func():
    pass


def dd_dynamic_tasks_map(func, ddf, name, **kwargs):
    n = ddf.npartitions
    df_name = ddf._name
    token = tokenize(ddf, func, name)
    name = "%s-%s" % (name, token)
    rearguard_name = "rearguard_" + name

    layer = {}
    for rank in range(n):
        layer[(name, rank)] = (
            dask.utils.apply,
            func,
            [(df_name, rank), rank, n, name],
            kwargs,
        )

    ddf = new_dd_object(
        HighLevelGraph.from_collections(name, layer, dependencies=[ddf]),
        name,
        ddf._meta,
        ddf.divisions,
    )

    layer = {}
    for rank in range(n):
        layer[(rearguard_name, rank)] = (_dummy_func, (name, rank))

    ddf = new_dd_object(
        HighLevelGraph.from_collections(rearguard_name, layer, dependencies=[ddf]),
        rearguard_name,
        ddf._meta,
        ddf.divisions,
    )
    return ddf


def dynshuffle_kernel(df, rank, npartitions, name, col, ignore_index):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    assert name in myself

    print(
        f"[{worker.address}] dynshuffle_kernel() - myself: {repr(myself)}, rank: {rank}/{npartitions-1}, name: {name}, df: {type(df)}"
    )

    groups = shuffle_group(
        df,
        col,
        0,
        npartitions,
        npartitions,
        ignore_index=ignore_index,
        nfinal=npartitions,
    )
    assert len(groups) == npartitions

    new_tasks = []
    for i in range(npartitions):
        part = str((name, i))
        new_tasks.append(
            {
                "key": f"shuffle_getitem_{myself}_{part}",
                "deps": [part],
                "task": dumps_task((getitem, part, rank)),
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
        rearguard_key=str(("rearguard_" + name, rank)),
        rearguard_input=f"shuffle_join_{myself}",
    )

    return groups


def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    # print(f"rearrange_by_column_dynamic_tasks() - column: {column}, \nddf: {df.compute()}")
    df2 = dd_dynamic_tasks_map(
        dynshuffle_kernel, df, "dynshuffle", col=column, ignore_index=ignore_index,
    )

    if npartitions is not None and npartitions != df.npartitions:
        token = tokenize(df2, npartitions)
        repartition_group_token = "repartition-group-" + token

        dsk = {
            (repartition_group_token, i): (
                shuffle_group_2,
                k,
                column,
                ignore_index,
                npartitions,
            )
            for i, k in enumerate(df2.__dask_keys__())
        }

        repartition_get_name = "repartition-get-" + token

        for p in range(npartitions):
            dsk[(repartition_get_name, p)] = (
                shuffle_group_get,
                (repartition_group_token, p % df.npartitions),
                p,
            )

        graph2 = HighLevelGraph.from_collections(
            repartition_get_name, dsk, dependencies=[df2]
        )
        df3 = new_dd_object(
            graph2, repartition_get_name, df2._meta, [None] * (npartitions + 1)
        )
    else:
        df3 = df2
        df3.divisions = (None,) * (df.npartitions + 1)

    return df3
