"""谱系评估：沿固定父引用链上行，判定可交付状态并给出阻断路径。

读取与计算策略：
- 单件查询（/api/products/{id}/status）使用按需视图，并在本次评估内缓存
  已查过的修订/现役修订，避免同一条链上重复访问数据库。
- 批量查询（/api/mosaics）先在一个快照事务内一次性装载全部修订，
  之后所有拼图的评估均为纯内存计算；共享父链只读取一次，
  数据库访问次数不随拼图数量增长。
- 链路结论以链式结构缓存：每个节点只保存自身父边的阻断与一个指向父节点
  结论的引用（O(1)），共享父链因此只遍历一次；阻断路径只在为具体拼图
  输出结果时按该拼图起点物化，保证每条路径都从拼图自身出发。
  单条深链的内存占用为 O(链长)，与逐节点实现一致。
- 环的闭环锚点因起点而异：环路共享一份结构，各锚点按需惰性物化。
  单件查询只展开自身所在锚点（O(环长)）；批量列出环上全部拼图时
  输出本身即为 O(环长²) 条路径。
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Union

from . import database as db
from .schemas import (
    BLOCKER_LABELS,
    REASON_CYCLE,
    REASON_MISSING,
    REASON_SUPERSEDED,
)

NodeKey = tuple[str, int]
Blocker = dict[str, Any]
EMPTY_KEYS: frozenset[NodeKey] = frozenset()


def _node_label(product_id: str, revision: int) -> str:
    return f"{product_id}@{revision}"


def _key_label(key: NodeKey) -> str:
    return _node_label(key[0], key[1])


class DbLineageView:
    """按需读取的谱系视图：单次评估内缓存修订行与现役行。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._revisions: dict[NodeKey, Optional[sqlite3.Row]] = {}
        self._active: dict[str, Optional[sqlite3.Row]] = {}
        self.memo: dict[NodeKey, "_ChainMemo"] = {}

    def get_revision(
        self, product_id: str, revision: int
    ) -> Optional[sqlite3.Row]:
        key = (product_id, revision)
        if key not in self._revisions:
            self._revisions[key] = db.get_revision(
                self._conn, product_id, revision
            )
        return self._revisions[key]

    def get_active_revision(self, product_id: str) -> Optional[sqlite3.Row]:
        if product_id not in self._active:
            self._active[product_id] = db.get_active_revision(
                self._conn, product_id
            )
        return self._active[product_id]


class SnapshotLineageView:
    """预装载的谱系视图：构造后再评估不产生任何数据库访问。"""

    def __init__(self, rows: list[sqlite3.Row]) -> None:
        revisions: dict[NodeKey, sqlite3.Row] = {}
        active: dict[str, sqlite3.Row] = {}
        for row in rows:
            product_id = row["product_id"]
            revision = row["revision"]
            revisions[(product_id, revision)] = row
            current = active.get(product_id)
            if current is None or revision > current["revision"]:
                active[product_id] = row
        self._revisions = revisions
        self._active = active
        # 跨拼图共享：同一节点的链路结论只构建一次
        self.memo: dict[NodeKey, "_ChainMemo"] = {}

    def get_revision(
        self, product_id: str, revision: int
    ) -> Optional[sqlite3.Row]:
        return self._revisions.get((product_id, revision))

    def get_active_revision(self, product_id: str) -> Optional[sqlite3.Row]:
        return self._active.get(product_id)

    def active_products(self) -> list[sqlite3.Row]:
        """全部成果号的现役修订行，按成果号排序。"""
        return [self._active[pid] for pid in sorted(self._active)]


class _ChainLink:
    """某节点相对其自身的链路结论（不可变，可被多个下游共享）。

    edge：该节点父边的 superseded 阻断模板（不含从起点来的路径前缀）。
    parent：父节点的结论；缺失父项或链到顶时 parent 为 None。
    terminals：仅链终止节点使用，记录 ok / missing。
    has_blockers：该节点及上游是否存在任何阻断，用于干净链的 O(1) 判定。
    cycle_keys：上游环上的节点集合引用（通常为空），用于判断缓存能否复用。
    """

    __slots__ = ("key", "edge", "parent", "terminals", "has_blockers", "cycle_keys")

    def __init__(self, key: NodeKey) -> None:
        self.key = key
        self.edge: Optional[Blocker] = None
        self.parent: Optional["_ChainMemo"] = None
        self.terminals: Optional[dict[str, Any]] = None
        self.has_blockers = False
        self.cycle_keys: frozenset[NodeKey] = EMPTY_KEYS


class _CycleMemo:
    """一个环路的共享结论：环上节点的顺序与各父边的退役阻断。

    环的闭环锚点因起点而异，因此按锚点惰性物化并缓存。
    单件查询只物化自己所在的锚点（O(环长)）；批量列出环上全部拼图时，
    输出本身就是 O(环长²) 条路径，此时才逐个锚点展开。
    """

    __slots__ = ("nodes", "edges", "cycle_keys", "has_blockers", "_anchors")

    def __init__(
        self, nodes: list[NodeKey], edges: list[Optional[Blocker]]
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.cycle_keys = frozenset(nodes)
        self.has_blockers = True
        self._anchors: dict[int, list[Blocker]] = {}

    def blockers_from(self, index: int) -> list[Blocker]:
        cached = self._anchors.get(index)
        if cached is not None:
            return cached

        length = len(self.nodes)
        path = [_key_label(self.nodes[index])]
        blockers: list[Blocker] = []
        for step in range(length):
            source = (index + step) % length
            nxt = (index + step + 1) % length
            path.append(_key_label(self.nodes[nxt]))
            if nxt == index:
                # 回到起点：与逐节点评估一致，先判环、不再检查退役
                blockers.append(_cycle_blocker(list(path)))
                break
            if self.edges[source] is not None:
                blockers.append({**self.edges[source], "path": list(path)})
        self._anchors[index] = blockers
        return blockers


class _CycleAnchor:
    """指向环路中某个锚点的轻量引用（可被多个下游共享）。"""

    __slots__ = ("cycle", "index", "has_blockers", "cycle_keys")

    def __init__(self, cycle: _CycleMemo, index: int) -> None:
        self.cycle = cycle
        self.index = index
        self.has_blockers = True
        self.cycle_keys = cycle.cycle_keys

    def blockers(self) -> list[Blocker]:
        return self.cycle.blockers_from(self.index)


_ChainMemo = Union["_ChainLink", _CycleAnchor]


def _missing_blocker(source_key: NodeKey, parent_key: NodeKey) -> Blocker:
    return {
        "reason": REASON_MISSING,
        "reason_label": BLOCKER_LABELS[REASON_MISSING],
        "path": [_key_label(source_key), _key_label(parent_key)],
        "detail": (
            f"成果 {_key_label(source_key)} 固定引用的父项 "
            f"{_key_label(parent_key)} 尚未提交"
        ),
    }


def _superseded_blocker(
    source_key: NodeKey, parent_key: NodeKey, active_revision: int
) -> Blocker:
    return {
        "reason": REASON_SUPERSEDED,
        "reason_label": BLOCKER_LABELS[REASON_SUPERSEDED],
        "path": [_key_label(source_key), _key_label(parent_key)],
        "detail": (
            f"被引用的父修订 {_key_label(parent_key)} 已非现役，"
            f"现役修订为 {_node_label(parent_key[0], active_revision)}"
        ),
    }


def _cycle_blocker(cycle_labels: list[str]) -> Blocker:
    return {
        "reason": REASON_CYCLE,
        "reason_label": BLOCKER_LABELS[REASON_CYCLE],
        "path": cycle_labels,
        "detail": f"引用链在 {cycle_labels[0]} 处形成闭环",
    }


def _edge_blocker(
    view: Any, source_key: NodeKey, parent_key: NodeKey
) -> Optional[Blocker]:
    """父项存在且非现役时返回 superseded 阻断（相对该父边）。"""
    active_row = view.get_active_revision(parent_key[0])
    if active_row is None or active_row["revision"] == parent_key[1]:
        return None
    return _superseded_blocker(
        source_key, parent_key, active_row["revision"]
    )


def _materialize(memo: _ChainMemo) -> list[Blocker]:
    """把相对某起点的链路结论展开为完整阻断路径（路径从起点出发）。"""
    if isinstance(memo, _CycleAnchor):
        return [{**b, "path": list(b["path"])} for b in memo.blockers()]
    if not memo.has_blockers:
        return []

    links: list[_ChainLink] = []
    node: Optional[_ChainMemo] = memo
    while isinstance(node, _ChainLink):
        links.append(node)
        node = node.parent

    labels: list[str] = []
    blockers: list[Blocker] = []
    for link in links:
        labels.append(_key_label(link.key))
        if link.edge is not None:
            # edge.path = [本节点, 父节点]；前缀已含本节点
            blockers.append({
                **link.edge,
                "path": [*labels, link.edge["path"][1]],
            })

    if isinstance(node, _CycleAnchor):
        for blocker in node.blockers():
            blockers.append({
                **blocker,
                "path": [*labels, *blocker["path"]],
            })
        return blockers

    terminal = links[-1].terminals
    if terminal is not None and terminal["kind"] == "missing":
        blockers.append({
            **terminal["blocker"],
            "path": [*labels, terminal["parent_label"]],
        })
    return blockers


def _memoize_cycle(
    stack: list[_ChainLink],
    cycle_start: int,
    view: Any,
    memo: dict[NodeKey, _ChainMemo],
) -> None:
    """为环路建立共享结构，环上各节点按锚点惰性物化。"""
    cycle = stack[cycle_start:]
    nodes = [link.key for link in cycle]
    length = len(nodes)

    # 环上每条父边都可能退役：统一按当前快照重算，避免漏算环入口那条边
    edges: list[Optional[Blocker]] = []
    for index in range(length):
        edges.append(_edge_blocker(view, nodes[index], nodes[(index + 1) % length]))

    cycle_memo = _CycleMemo(nodes, edges)
    for index, node in enumerate(nodes):
        memo[node] = _CycleAnchor(cycle_memo, index)


def _link_chain(
    stack: list[_ChainLink],
    memo: dict[NodeKey, _ChainMemo],
    parent: Optional[_ChainMemo],
) -> None:
    """把本次下行构造的链环自顶向下挂到 memo 上（只建引用，不复制路径）。

    parent 是栈顶节点已知的父结论（缺失父项/链到顶时为 None）。
    has_blockers / cycle_keys 由父结论 O(1) 传播，保证不读取过期值。
    """
    node: Optional[_ChainMemo] = parent
    for link in reversed(stack):
        link.parent = node
        if node is not None:
            link.has_blockers = link.edge is not None or node.has_blockers
            link.cycle_keys = node.cycle_keys
        else:
            link.has_blockers = link.edge is not None or (
                link.terminals is not None
                and link.terminals["kind"] == "missing"
            )
            link.cycle_keys = EMPTY_KEYS
        memo[link.key] = link
        node = link


def _reusable(memo: _ChainMemo, position: dict[NodeKey, int]) -> bool:
    """缓存结论可否直接挂接。

    同一条单父链上的节点不会重复出现，唯一风险是上游结论的环路已包含
    本次下行链上的节点：此时闭环锚点应换成链上更早的节点，必须继续
    下行让环检测自然发生。下行链通常很短，因此在 position 一侧检查。
    """
    if not memo.cycle_keys:
        return True
    return not any(key in memo.cycle_keys for key in position)


def _chain_memo(start_row: sqlite3.Row, view: Any) -> _ChainMemo:
    """取得从 start_row 出发的链路结论；共享部分只构建一次。"""
    start_key: NodeKey = (start_row["product_id"], start_row["revision"])
    cached = view.memo.get(start_key)
    if cached is not None:
        return cached

    memo = view.memo
    stack: list[_ChainLink] = []
    position: dict[NodeKey, int] = {}
    current = start_row

    while True:
        current_key: NodeKey = (current["product_id"], current["revision"])

        # 父节点已有结论且与本次链路无环冲突：直接挂接，共享部分不再遍历
        cached = memo.get(current_key)
        if cached is not None and _reusable(cached, position):
            if stack:
                _link_chain(stack, memo, cached)
            return memo[start_key]

        link = _ChainLink(current_key)
        position[current_key] = len(stack)
        stack.append(link)

        parent_pid = current["parent_product_id"]
        if parent_pid is None:
            link.terminals = {"kind": "ok"}
            _link_chain(stack, memo, None)
            return memo[start_key]

        parent_rev = current["parent_revision"]
        parent_key: NodeKey = (parent_pid, parent_rev)
        parent_row = view.get_revision(parent_pid, parent_rev)
        if parent_row is None:
            link.terminals = {
                "kind": "missing",
                "blocker": _missing_blocker(current_key, parent_key),
                "parent_label": _key_label(parent_key),
            }
            _link_chain(stack, memo, None)
            return memo[start_key]

        if parent_key in position:
            cycle_start = position[parent_key]
            _memoize_cycle(stack, cycle_start, view, memo)
            # 环上节点已是物化结论；环外祖先以它为父结论逐级挂接
            _link_chain(stack[:cycle_start], memo, memo[parent_key])
            return memo[start_key]

        link.edge = _edge_blocker(view, current_key, parent_key)
        current = parent_row


def evaluate_row(
    start_row: sqlite3.Row,
    *,
    view: Any,
    start_is_active: bool,
    materialize_blockers: bool = True,
) -> dict[str, Any]:
    """评估某个已存在修订行的整条谱系（只读视图，不直接访问数据库）。

    阻断分三类，沿引用路径逐层记录：
      - missing     ：父成果号/父修订号从未提交（父项允许晚到，晚到后重查即恢复）
      - superseded  ：父成果号存在更高已提交修订，被引用的旧修订已非现役
      - cycle       ：引用链回到链路上的某个节点
    一次评估会走完整条链并收集全部阻断，而不是遇错即停。

    materialize_blockers=False 时只判定可交付性、不展开路径，
    供“仅要计数/过滤、不返回明细”的批量场景省去路径拼装开销。
    """
    memo = _chain_memo(start_row, view)
    deliverable = not memo.has_blockers
    blockers = [] if (deliverable or not materialize_blockers) else _materialize(memo)
    kind = "tile" if start_row["parent_product_id"] is None else "mosaic"
    return {
        "product_id": start_row["product_id"],
        "revision": start_row["revision"],
        "is_active": start_is_active,
        "kind": kind,
        "summary": start_row["summary"],
        "parent_product_id": start_row["parent_product_id"],
        "parent_revision": start_row["parent_revision"],
        "deliverable": deliverable,
        "blockers": blockers,
    }


def evaluate(
    conn: sqlite3.Connection,
    start_row: sqlite3.Row,
    *,
    start_is_active: bool,
) -> dict[str, Any]:
    """兼容入口：以按需视图评估单个修订行。"""
    return evaluate_row(
        start_row,
        view=DbLineageView(conn),
        start_is_active=start_is_active,
    )


def evaluate_product(
    conn: sqlite3.Connection,
    product_id: str,
    revision: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """评估某成果号；revision 为空时评估现役（最高）修订。

    成果号不存在返回 None；指定修订不存在也返回 None（由路由层区分 404 文案）。
    """
    view = DbLineageView(conn)
    if revision is None:
        row = view.get_active_revision(product_id)
        if row is None:
            return None
        return evaluate_row(row, view=view, start_is_active=True)

    row = view.get_revision(product_id, revision)
    if row is None:
        return None
    active = view.get_active_revision(product_id)
    is_active = active is not None and active["revision"] == revision
    return evaluate_row(row, view=view, start_is_active=is_active)


def evaluate_active_mosaics(
    conn: sqlite3.Connection,
    deliverable: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """评估全部现役拼图，可按可交付结论过滤。

    调用方须在同一个快照事务内调用。整个批次只执行一次全量修订读取；
    共享父链的链路结论只构建一次，读取与遍历都不随拼图数量增长
    （输出路径本身的长度除外）。
    """
    view = SnapshotLineageView(db.list_all_revisions(conn))
    results: list[dict[str, Any]] = []
    for row in view.active_products():
        if row["parent_product_id"] is None:
            continue  # 只列拼图（有父引用的成果）
        memo = _chain_memo(row, view)
        is_deliverable = not memo.has_blockers
        if deliverable is not None and is_deliverable is not deliverable:
            # 过滤不命中：不展开路径，避免为丢弃的结果付出路径拼装成本
            continue
        if is_deliverable:
            blockers: list[Blocker] = []
        else:
            blockers = _materialize(memo)
        results.append({
            "product_id": row["product_id"],
            "revision": row["revision"],
            "is_active": True,
            "kind": "mosaic",
            "summary": row["summary"],
            "parent_product_id": row["parent_product_id"],
            "parent_revision": row["parent_revision"],
            "deliverable": is_deliverable,
            "blockers": blockers,
        })
    return results
