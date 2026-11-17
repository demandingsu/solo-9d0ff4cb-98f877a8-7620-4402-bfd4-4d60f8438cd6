"""批量状态查询专项自测。

覆盖：
1. /api/mosaics 的数据库访问次数与拼图数量无关（共享父链只读一次）。
2. 长链、父项缺失、旧修订失效、成环时，每个拼图仍给出从自身出发的完整路径。
3. 查询与提交交错：一次响应只对应同一已提交谱系，下一次查询看到新修订，
   失败（拒收）上传不改变既有结论。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app import database as db
from app import lineage
from app.main import create_app, get_conn


def _item(product_id, revision, summary=None, parent=None):
    return {
        "product_id": product_id,
        "revision": revision,
        "summary": summary or f"{product_id} v{revision}",
        "parent_product_id": parent[0] if parent else None,
        "parent_revision": parent[1] if parent else None,
    }


def _post(client, items):
    r = client.post("/api/batches", json={"items": items})
    assert r.status_code == 201, r.text
    return r.json()


@contextmanager
def _make_client(tmp_path, name="batch.db"):
    app = create_app(str(tmp_path / name))
    with TestClient(app) as c:
        yield c


def _build_shared_chain(client, depth: int, mosaics: int) -> None:
    """构造 depth 级父链，并让 mosaics 个拼图共享同一条链。"""
    _post(client, [_item("R", 1)])
    parent = ("R", 1)
    for level in range(1, depth + 1):
        pid = f"L{level}"
        _post(client, [_item(pid, 1, parent=parent)])
        parent = (pid, 1)
    _post(client, [
        _item(f"M{i}", 1, parent=parent) for i in range(mosaics)
    ])


# ---------------------------------------------------------------------------
# 1. 查询次数：共享父链不随拼图数量重复访问
# ---------------------------------------------------------------------------

def test_mosaic_listing_reads_revisions_once_regardless_of_mosaic_count(tmp_path):
    with _make_client(tmp_path) as client:
        _build_shared_chain(client, depth=12, mosaics=300)

        trace_sql: list[str] = []

        conn = db.connect(str(tmp_path / "batch.db"))
        conn.set_trace_callback(trace_sql.append)

        def traced_conn():
            # 复用同一连接即可：只统计本次请求内的 SQL
            conn.execute("PRAGMA query_only=ON")
            try:
                yield conn
            finally:
                conn.execute("PRAGMA query_only=OFF")

        client.app.dependency_overrides[get_conn] = traced_conn
        try:
            r = client.get("/api/mosaics")
        finally:
            client.app.dependency_overrides.pop(get_conn, None)
            conn.close()

        assert r.status_code == 200
        assert len(r.json()) == 12 + 300

        reads = [
            sql for sql in trace_sql
            if sql.lstrip().upper().startswith("SELECT")
            and "FROM revisions" in sql
        ]
        # 全量修订只读取一次，不随 300 个拼图或 12 级共享父链增加
        assert len(reads) == 1, reads


def test_mosaic_listing_does_not_use_per_node_queries(tmp_path, monkeypatch):
    with _make_client(tmp_path) as client:
        _build_shared_chain(client, depth=8, mosaics=50)

        def _forbidden(*args, **kwargs):
            raise AssertionError("批量查询不应逐节点访问数据库")

        monkeypatch.setattr(db, "get_revision", _forbidden)
        monkeypatch.setattr(db, "get_active_revision", _forbidden)
        r = client.get("/api/mosaics")
        assert r.status_code == 200
        ids = [m["product_id"] for m in r.json()]
        assert len(ids) == 8 + 50  # 8 级链节点 + 50 个末端拼图
        assert ids.count("M0") == 1


# ---------------------------------------------------------------------------
# 2. 每个拼图给出从自身出发的完整阻断路径
# ---------------------------------------------------------------------------

def test_each_mosaic_has_own_complete_blocker_path(tmp_path):
    with _make_client(tmp_path) as client:
        # 共享链：M1/M2/M3 -> C -> B -> A，再向上缺失
        _post(client, [
            _item("A", 1, parent=("MISSING", 7)),
            _item("B", 1, parent=("A", 1)),
            _item("C", 1, parent=("B", 1)),
            _item("M1", 1, parent=("C", 1)),
            _item("M2", 1, parent=("C", 1)),
            _item("M3", 1, parent=("C", 1)),
        ])
        by_id = {m["product_id"]: m for m in client.get("/api/mosaics").json()}
        for pid in ("A", "B", "C", "M1", "M2", "M3"):
            mosaic = by_id[pid]
            assert mosaic["deliverable"] is False
            blocker = mosaic["blockers"][0]
            assert blocker["reason"] == "missing"
            assert blocker["path"][0] == f"{pid}@1"
            assert blocker["path"][-1] == "MISSING@7"
        assert by_id["M1"]["blockers"][0]["path"] == [
            "M1@1", "C@1", "B@1", "A@1", "MISSING@7"
        ]
        assert by_id["M2"]["blockers"][0]["path"][0] == "M2@1"


def test_long_chain_missing_superseded_cycle_batch(tmp_path):
    depth = 1200
    with _make_client(tmp_path) as client:
        # 深链：L1 <- L2 <- ... <- L1200，底部父项缺失
        items = [_item("L1", 1, parent=("MISSING", 1))]
        for level in range(2, depth + 1):
            items.append(_item(f"L{level}", 1, parent=(f"L{level-1}", 1)))
        _post(client, items)

        # 退役链：S 有 v1/v2，E 固定引用 S@1
        _post(client, [_item("S", 1), _item("S", 2)])
        _post(client, [_item("E", 1, parent=("S", 1))])

        # 环：X@1 -> Y@1 -> Z@1 -> X@1
        _post(client, [
            _item("X", 1, parent=("Y", 1)),
            _item("Y", 1, parent=("Z", 1)),
            _item("Z", 1, parent=("X", 1)),
        ])

        listing = client.get("/api/mosaics", params={"deliverable": False}).json()
        by_id = {m["product_id"]: m for m in listing}

        top = by_id[f"L{depth}"]
        assert top["blockers"][0]["reason"] == "missing"
        assert len(top["blockers"][0]["path"]) == depth + 1
        assert top["blockers"][0]["path"][0] == f"L{depth}@1"
        assert top["blockers"][0]["path"][-1] == "MISSING@1"

        assert by_id["E"]["blockers"][0]["reason"] == "superseded"
        assert by_id["E"]["blockers"][0]["path"] == ["E@1", "S@1"]

        for pid in ("X", "Y", "Z"):
            blocker = by_id[pid]["blockers"][0]
            assert blocker["reason"] == "cycle"
            assert blocker["path"][0] == f"{pid}@1"
            assert blocker["path"][-1] == f"{pid}@1"
            assert len(blocker["path"]) == 4

        # 单件接口与历史修订接口的结论保持一致
        single = client.get(f"/api/products/L{depth}/status").json()
        assert single["blockers"] == top["blockers"]


def test_cycle_entry_from_each_node_has_own_path_and_anchor(tmp_path):
    with _make_client(tmp_path) as client:
        # 环 A->B->C->A，外部 D->C，另有一环内退役引用 B@2
        _post(client, [
            _item("A", 1, parent=("B", 1)),
            _item("B", 1, parent=("C", 1)),
            _item("C", 1, parent=("A", 1)),
            _item("D", 1, parent=("C", 1)),
            _item("B", 2),
        ])
        listing = {m["product_id"]: m for m in client.get("/api/mosaics").json()}

        expectations = {
            "A": ["A@1", "B@1", "C@1", "A@1"],
            "B": ["B@1", "C@1", "A@1", "B@1"],
            "C": ["C@1", "A@1", "B@1", "C@1"],
            "D": ["D@1", "C@1", "A@1", "B@1", "C@1"],
        }
        for pid, expected_path in expectations.items():
            if pid in ("A", "B"):
                # 现役修订更高，需经单件接口查看旧修订 A@1 / B@1
                entry = client.get(
                    f"/api/products/{pid}/status", params={"revision": 1}
                ).json()
            else:
                entry = listing[pid]
            cycle_blockers = [
                b for b in entry["blockers"] if b["reason"] == "cycle"
            ]
            assert cycle_blockers, pid
            # 无论入口在哪，环的完整路径都必须从该拼图自身出发
            assert cycle_blockers[-1]["path"] == expected_path, pid
            # 闭环说明：环上节点锚定自身；环外入口锚定它进入环的节点
            anchor = pid if pid != "D" else "C"
            assert cycle_blockers[-1]["detail"].endswith(
                f"{anchor}@1 处形成闭环"
            )

        # 本次请求内多次评估同一环，结论必须稳定
        again = {m["product_id"]: m for m in client.get("/api/mosaics").json()}
        assert again == listing


def test_cycle_with_superseded_edge_reports_both(tmp_path):
    with _make_client(tmp_path) as client:
        _post(client, [
            _item("A", 1, parent=("B", 1)),
            _item("B", 1, parent=("A", 1)),
            _item("A", 2, parent=("B", 1)),  # A@2 现役，A@1 退役
        ])
        listing = {m["product_id"]: m for m in client.get("/api/mosaics").json()}
        # B@1 的父边指向已退役的 A@1，同时 A@1->B@1 成环
        reasons = [b["reason"] for b in listing["B"]["blockers"]]
        assert reasons == ["superseded", "cycle"]
        assert listing["B"]["blockers"][0]["path"] == ["B@1", "A@1"]
        assert listing["B"]["blockers"][1]["path"] == ["B@1", "A@1", "B@1"]
        # 现役 A@2 自身不满足环条件（它引用 B@1，B@1 引用 A@1）
        assert listing["A"]["deliverable"] is False
        assert listing["A"]["blockers"][0]["reason"] == "superseded"


def test_listing_work_does_not_grow_with_mosaic_count(tmp_path, monkeypatch):
    """共享父链的链路构建次数按父链节点数计，不按拼图数计。"""
    calls = {"n": 0}
    original = lineage._chain_memo

    def counted(row, view):
        key = (row["product_id"], row["revision"])
        before = view.memo.get(key)
        result = original(row, view)
        if before is None and view.memo.get(key) is not None:
            calls["n"] += 1
        return result

    monkeypatch.setattr(lineage, "_chain_memo", counted)
    with _make_client(tmp_path) as client:
        _build_shared_chain(client, depth=30, mosaics=400)
        client.get("/api/mosaics")
        # 30 级链节点各构建一次 + 400 个末端拼图各一次（每个拼图自身是新节点）
        assert calls["n"] <= 30 + 400 + 1
        # 远小于“按拼图数 × 链深”的重复遍历量
        assert calls["n"] < 400 * 30


# ---------------------------------------------------------------------------
# 3. 查询与提交交错
# ---------------------------------------------------------------------------

def test_query_snapshot_is_consistent_then_next_query_sees_update(tmp_path, monkeypatch):
    db_path = str(tmp_path / "interleave.db")
    app = create_app(db_path)
    with TestClient(app) as client:
        _post(client, [_item("R", 1), _item("M", 1, parent=("R", 1))])
        assert client.get("/api/mosaics").json()[0]["deliverable"] is True

        # 在批量评估装载完快照、尚未返回前，另一连接提交 R@2
        original_init = lineage.SnapshotLineageView.__init__
        state = {"written": False}

        def init_with_concurrent_commit(self, rows):
            original_init(self, rows)
            if not state["written"]:
                state["written"] = True
                writer = db.connect(db_path)
                try:
                    result = db.submit_batch(writer, [_item("R", 2)])
                    assert result["status"] == "accepted"
                finally:
                    writer.close()

        monkeypatch.setattr(
            lineage.SnapshotLineageView, "__init__", init_with_concurrent_commit
        )
        first = client.get("/api/mosaics").json()
        monkeypatch.setattr(
            lineage.SnapshotLineageView, "__init__", original_init
        )

        # 本次响应仍对应装载时的同一已提交谱系：M@1 仍可交付
        assert [m["product_id"] for m in first] == ["M"]
        assert first[0]["deliverable"] is True
        assert first[0]["blockers"] == []

        # 下一次查询必须看到新修订：M@1 引用 R@1，R@1 已退役
        second = client.get("/api/mosaics").json()
        assert second[0]["deliverable"] is False
        assert second[0]["blockers"][0]["reason"] == "superseded"
        assert second[0]["blockers"][0]["path"] == ["M@1", "R@1"]
        assert "R@2" in second[0]["blockers"][0]["detail"]


def test_concurrent_submit_between_queries_is_observed_cleanly(tmp_path):
    """列表查询与真实并发提交交错：每次响应自洽，下一次查询看到新修订。"""
    db_path = str(tmp_path / "threaded.db")
    app = create_app(db_path)
    with TestClient(app) as client:
        _post(client, [_item("R", 1), _item("M", 1, parent=("R", 1))])

        observed: list[bool] = []
        errors: list[Exception] = []
        started = threading.Event()
        written = threading.Event()

        def writer():
            try:
                started.wait(timeout=5)
                with TestClient(app) as w:
                    _post(w, [_item("R", 2)])
                written.set()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=writer)
        t.start()
        try:
            started.set()
            for _ in range(40):
                listing = client.get("/api/mosaics").json()
                assert len(listing) == 1
                mosaic = listing[0]
                # 单次响应内部必须自洽：可交付 ⇔ 无阻断
                assert mosaic["deliverable"] == (mosaic["blockers"] == [])
                if not mosaic["deliverable"]:
                    assert mosaic["blockers"][0]["reason"] == "superseded"
                    assert mosaic["blockers"][0]["path"] == ["M@1", "R@1"]
                observed.append(mosaic["deliverable"])
        finally:
            t.join()

        assert errors == []
        assert written.is_set()
        assert observed[0] is True  # 提交前：R@1 现役
        assert observed[-1] is False  # 提交后：R@1 退役
        final = client.get("/api/mosaics").json()[0]
        assert final["deliverable"] is False
        assert "R@2" in final["blockers"][0]["detail"]


def test_rejected_upload_does_not_change_mosaic_conclusion(tmp_path):
    with _make_client(tmp_path) as client:
        _post(client, [_item("R", 1), _item("M", 1, parent=("R", 1))])
        before = client.get("/api/mosaics").json()

        rejected = _post(client, [
            _item("R", 1, summary="篡改摘要"),
            _item("NEW", 1),
        ])
        assert rejected["status"] == "rejected"
        assert rejected["inserted_count"] == 0

        after = client.get("/api/mosaics").json()
        assert after == before
        assert client.get("/api/products/NEW/status").status_code == 404
        assert client.get("/api/products/R/status").json()["summary"] == "R v1"


def test_listing_transaction_rolls_back_on_error(tmp_path):
    """读取中途异常时快照事务必须回滚，连接仍可继续使用。"""
    with _make_client(tmp_path) as client:
        _post(client, [_item("R", 1), _item("M", 1, parent=("R", 1))])

        original = lineage.evaluate_active_mosaics

        def boom(*args, **kwargs):
            raise RuntimeError("评估失败")

        lineage.evaluate_active_mosaics = boom
        try:
            with pytest.raises(RuntimeError):
                client.get("/api/mosaics")
        finally:
            lineage.evaluate_active_mosaics = original

        # 后续查询正常，说明事务未悬挂
        assert client.get("/api/mosaics").json()[0]["deliverable"] is True


def test_snapshot_rolls_back_and_connection_stays_usable(tmp_path):
    conn = db.connect(str(tmp_path / "snap.db"))
    db.init_db(conn)
    try:
        with pytest.raises(RuntimeError):
            with db.snapshot(conn):
                db.list_all_revisions(conn)
                raise RuntimeError("boom")
        # 回滚后不在事务中，连接可立即开启新事务
        assert not conn.in_transaction
        with db.snapshot(conn):
            assert db.list_all_revisions(conn) == []
    finally:
        conn.close()
