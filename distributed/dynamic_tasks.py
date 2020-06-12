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


def _rearguard():
    pass


def dynshuffle_kernel(
    df, rank, npartitions, name, max_branch, col, ignore_index, nfinal, meta_tasks
):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    assert name in myself

    groups = shuffle_group(
        df, col, 0, npartitions, npartitions, ignore_index=ignore_index, nfinal=nfinal,
    )
    assert len(groups) == npartitions

    new_tasks = []
    for i in range(npartitions):
        part = str((name, i))
        new_tasks.append(
            {
                "key": f"shuffle_getitem_{myself}_{part}",
                "dependencies": [part],
                "task": dumps_task((getitem, part, rank)),
                "priority": 0,
            }
        )
    getitem_keys = [t["key"] for t in new_tasks]

    new_tasks.append(
        {
            "key": f"shuffle_join_{myself}",
            "dependencies": getitem_keys,
            "task": dumps_task((_concat, getitem_keys)),
        }
    )

    client.sync(
        worker.scheduler.insert_tasks,
        cur_key=myself,
        new_tasks=new_tasks,
        rearguard_key=str(("rearguard_" + name, rank)),
        rearguard_input=f"shuffle_join_{myself}",
    )

    return groups


import math
from dask.utils import digit, insert


def _noop(x):
    return x


def create_tasks_for_input(
    inp,
    token,
    n,
    k,
    stages,
    myself,
    worker_address,
    rank,
    npartitions,
    inputs,
    col,
    ignore_index,
    name,
    nfinal,
    meta_task,
):

    shuffle_join_name = "shuffle-join-" + token
    shuffle_group_name = "shuffle-group-" + token
    shuffle_split_name = "shuffle-split-" + token
    shuffle_token = "shuffle-" + token

    # print(
    #     f"[{worker_address}] kernel() - myself: {repr(myself)}, rank: {rank}/{npartitions-1}, k:{k}, inputs: {len(inputs)}, token: {token}"
    # )

    new_tasks = []
    if rank < npartitions:
        new_tasks.append(
            {
                "key": str((shuffle_join_name, 0, inp)),
                "dependencies": [myself],
                "task": dumps_task((_noop, myself)),
            }
        )
    else:
        new_tasks.append(
            {
                "key": str((shuffle_join_name, 0, inp)),
                "dependencies": [myself],
                "task": meta_task,
            }
        )

    for stage in range(1, stages + 1):
        new_tasks.append(
            {
                "key": str((shuffle_group_name, stage, inp)),
                "dependencies": [str((shuffle_join_name, stage - 1, inp))],
                "task": dumps_task(
                    (
                        shuffle_group,
                        str((shuffle_join_name, stage - 1, inp)),
                        col,
                        stage - 1,
                        k,
                        n,
                        ignore_index,
                        nfinal,
                    )
                ),
            }
        )
        _concat_list = []
        _deps = []
        _dependents = []
        for i in range(k):
            getitem_key = insert(inp, stage - 1, i)
            new_tasks.append(
                {
                    "key": str((shuffle_split_name, stage, i, inp)),
                    "dependencies": [str((shuffle_group_name, stage, inp))],
                    "dependents": [str((shuffle_join_name, stage, getitem_key))],
                    "task": dumps_task(
                        (getitem, str((shuffle_group_name, stage, inp)), i)
                    ),
                }
            )
            key = str((shuffle_split_name, stage, inp[stage - 1], getitem_key))
            _concat_list.append(key)
            _deps.extend([(key, str((name, inputs.index(getitem_key))))])

        new_tasks.append(
            {
                "key": str((shuffle_join_name, stage, inp)),
                "dependencies": _deps,
                "task": dumps_task((_concat, _concat_list, ignore_index)),
            }
        )
    return new_tasks


def dynshuffle_staging_kernel(
    df, rank, npartitions, name, max_branch, col, ignore_index, nfinal, meta_task
):
    worker = get_worker()
    client = get_client()
    myself = worker.get_current_task()
    assert name in myself
    n = npartitions
    token = name.rsplit("-", 1)[1]

    shuffle_join_name = "shuffle-join-" + token

    stages = int(math.ceil(math.log(n) / math.log(max_branch)))
    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n

    inputs = [tuple(digit(i, j, k) for j in range(stages)) for i in range(k ** stages)]
    if rank == npartitions - 1:
        new_tasks = []
        for i, inp in enumerate(inputs[rank:]):
            new_tasks.extend(
                create_tasks_for_input(
                    inp,
                    token,
                    n,
                    k,
                    stages,
                    myself,
                    worker.address,
                    rank + i,
                    npartitions,
                    inputs,
                    col,
                    ignore_index,
                    name,
                    nfinal,
                    meta_task
                )
            )
    else:
        new_tasks = create_tasks_for_input(
            inputs[rank],
            token,
            n,
            k,
            stages,
            myself,
            worker.address,
            rank,
            npartitions,
            inputs,
            col,
            ignore_index,
            name,
            nfinal,
            meta_task
        )

    client.sync(
        worker.scheduler.insert_tasks,
        cur_key=myself,
        new_tasks=new_tasks,
        rearguard_key=str(("rearguard_" + name, rank)),
        rearguard_input=str((shuffle_join_name, stages, inputs[rank])),
    )

    return df


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
        layer[(rearguard_name, rank)] = (_rearguard, (name, rank))

    ddf = new_dd_object(
        HighLevelGraph.from_collections(rearguard_name, layer, dependencies=[ddf]),
        rearguard_name,
        ddf._meta,
        ddf.divisions,
    )
    return ddf


def rearrange_by_column_dynamic_tasks(
    df, column, max_branch=None, npartitions=None, ignore_index=False
):
    df2 = dd_dynamic_tasks_map(
        dynshuffle_staging_kernel,
        df,
        "dynshuffle",
        max_branch=max_branch if max_branch else 32,
        col=column,
        ignore_index=ignore_index,
        nfinal=npartitions if npartitions else df.npartitions,
        meta_task = dumps_task(df._meta)
    )

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
