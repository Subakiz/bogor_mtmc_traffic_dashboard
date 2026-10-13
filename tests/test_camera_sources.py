"""Extra camera sources: the Tasikmalaya adapter and how extra nodes join topology and database."""
import json
import os
import sqlite3

from backend import camera_sources as cs
from backend.database import Database
from backend.topology_engine import TopologyEngine

PAGE = 'x\nvar cctv=' + json.dumps({
    "5": {"id": "5", "nama": "Simpang Alun-alun Arah Otista", "jenis": "FIX", "link": "https://h/camera/a.m3u8",
          "lokasi_lat": "-7.3268", "lokasi_lng": "108.2248"},
    "6": {"id": "6", "nama": "Simpang Alun-alun Arah Sutisna", "jenis": "FIX", "link": "https://h/camera/b.m3u8",
          "lokasi_lat": "-7.3268", "lokasi_lng": "108.2247"},
    "7": {"id": "7", "nama": "Simpang Jati PTZ", "jenis": "PTZ", "link": "https://h/camera/c.m3u8",
          "lokasi_lat": "-7.30", "lokasi_lng": "108.20"},
}) + ';\nvar lokasi=[];'


def _source_dir(tmp_path):
    d = tmp_path / "sources"
    d.mkdir()
    (d / "tasik.json").write_text(json.dumps({"nodes": cs.parse_tasikmalaya(PAGE)}))
    return str(d)


def test_parse_namespaces_ids_and_groups_junctions():
    nodes = cs.parse_tasikmalaya(PAGE)
    assert [n["id"] for n in nodes] == [1005, 1006, 1007]          # clear of Bogor's 75-209
    assert nodes[0]["junction_name"] == nodes[1]["junction_name"] == "Simpang Alun-alun"
    assert nodes[2]["junction_name"] == "Simpang Jati" and nodes[2]["camera_type"] == "PTZ"
    assert all(n["stream_url"].startswith("https://") for n in nodes)


def test_loader_respects_env_switch(tmp_path, monkeypatch):
    d = _source_dir(tmp_path)
    assert len(cs.load_extra_nodes(d)) == 3                          # explicit dir ignores the switch
    monkeypatch.setattr(cs, "SOURCES_DIR", d)
    monkeypatch.setenv("EXTRA_CAMERA_SOURCES", "0")
    assert cs.load_extra_nodes() == []
    monkeypatch.setenv("EXTRA_CAMERA_SOURCES", "1")
    assert len(cs.load_extra_nodes()) == 3


def test_engine_and_database_include_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SOURCES_DIR", _source_dir(tmp_path))
    monkeypatch.setenv("EXTRA_CAMERA_SOURCES", "1")
    eng = TopologyEngine()
    assert eng.cameras[1005]["corridor"] == "TASIKMALAYA"
    assert eng.get_intra_junction_cluster(1005) == eng.get_intra_junction_cluster(1006)
    assert 1007 not in eng.get_intra_junction_cluster(1005)
    assert not eng.outgoing_edges[1005]
    path = str(tmp_path / "t.sqlite")
    Database(path, auto_seed=True)
    ids = {r[0] for r in sqlite3.connect(path).execute("SELECT id FROM cameras")}
    assert {1005, 1006, 1007} <= ids and 75 in ids
