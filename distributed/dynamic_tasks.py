from dask.base import tokenize
import dask.dataframe as dd
from distributed import get_worker, get_client
from distributed.worker import dumps_task
import dask
from dask.dataframe.shuffle import (
    shuffle,
    shuffle_group,
    _concat,
    shuffle_group_2,
    shuffle_group_get,
)
from dask.dataframe.core import new_dd_object
from operator import getitem
from dask.highlevelgraph import HighLevelGraph
import math
from dask.utils import digit, insert


def _rearguard():
    pass


def _getitem(x, y):
    print(f"getitem x: {type(x)}, y: {type(y)}")
    return getitem(x, y)


def __concat(args):
    print("concat: ", [type(a) for a in args])
    return _concat(args)


def dynshuffle_kernel(
    df,
    name,
    token,
    inp,
    stage,
    stages,
    rank,
    n,
    k,
    nfinal,
    max_branch,
    col,
    ignore_index,
):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    assert name in myself

    shuffle_getitem_name = name + "-getitem-" + token
    shuffle_concat_name = name + "-concat-" + token

    #print(
    #    f"[{worker.address}] kernel() - myself: {repr(myself)}, rank: {rank}/{n}, k:{k}, inp: {inp}, stage: {stage}, token: {token}"
    #)

    groups = shuffle_group(df, col, stage, k, n, ignore_index, nfinal)

    new_tasks = []
    for i in range(k):
        getitem_key = insert(inp, stage, i)
        new_tasks.append(
            {
                "key": str((shuffle_getitem_name, stage, rank, i)),
                "dependencies": [str((f"{name}-{token}", stage, getitem_key))],
                "task": dumps_task(
                    (_getitem, str((f"{name}-{token}", stage, getitem_key)), inp[stage])
                ),
                "priority": 0,
            }
        )
    getitem_keys = [t["key"] for t in new_tasks]

    new_tasks.append(
        {
            "key": str((shuffle_concat_name, stage, rank)),
            "dependencies": getitem_keys,
            "task": dumps_task((__concat, getitem_keys)),
        }
    )

    client.sync(
        worker.scheduler.insert_tasks,
        cur_key=myself,
        new_tasks=new_tasks,
        rearguard_key=str((f"rearguard_{name}_{stage}-{token}", rank)),
        rearguard_input=str((shuffle_concat_name, stage, rank)),
    )

    return groups


def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=None, npartitions=None, ignore_index=False
):
    #max_branch = 2
    token = tokenize(df, column, max_branch, npartitions, ignore_index)
    max_branch = max_branch if max_branch else 32
    n = df.npartitions
    nfinal = npartitions if npartitions else n

    stages = int(math.ceil(math.log(n) / math.log(max_branch)))
    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n

    inputs = [tuple(digit(i, j, k) for j in range(stages)) for i in range(k ** stages)]

    name = "dynshuffle"

    dsk = {}
    for stage in range(stages):
        for rank, inp in enumerate(inputs):
            if stage == 0:
                if rank < df.npartitions:
                    start = (df._name, rank)
                else:
                    start = df._meta
                    #start = (df._name, df.npartitions-1)
            else:
                start = (f"rearguard_{name}_{stage-1}-{token}", rank)
            #print((f"{name}-{token}", stage, inp))
            dsk[(f"{name}-{token}", stage, inp)] = (
                dynshuffle_kernel,
                start,
                name,
                token,
                inp,
                stage,
                stages,
                rank,
                n,
                k,
                nfinal,
                max_branch,
                column,
                ignore_index,
            )
            if stage == stages-1 and rank == df.npartitions-1:
                dsk[(f"rearguard_{name}_{stage}-{token}", rank)] = (
                    _rearguard,
                    [(f"{name}-{token}", stage, inn) for inn in inputs[rank:]] +
                    [(f"rearguard_{name}_{stage}-{token}", r) for r in range(rank+1, len(inputs))]
                )
            else:
                dsk[(f"rearguard_{name}_{stage}-{token}", rank)] = (
                    _rearguard,
                    (f"{name}-{token}", stage, inp),
                )


    df2 = new_dd_object(
        HighLevelGraph.from_collections(
            f"rearguard_{name}_{stages-1}-{token}", dsk, dependencies=[df]
        ),
        f"rearguard_{name}_{stages-1}-{token}",
        df._meta,
        df.divisions,
    )

    # df2.visualize(filename=f"graph2-{token}.svg")
    # df2.compute(optimize_graph=False)
    # assert False

    # If the npartitions doesn't match, we use the old shuffle code for now.
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
