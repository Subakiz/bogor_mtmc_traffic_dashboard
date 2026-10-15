"""
tests/test_scrape_sync.py
The scraper's topology/DB sync: a detail page that failed to load (status "error: ...") says nothing
about the stream, so it must not overwrite a working stream URL with None / "".
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import scrape_cctv_streams as scrape

URL = "https://restreamer.example.test/memfs/abc.m3u8"


class TestSyncKeepsStreamOnFetchError(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.topo = os.path.join(self.tmp.name, "topology.json")
        self.db = os.path.join(self.tmp.name, "traffic.sqlite")
        with open(self.topo, "w", encoding="utf-8") as f:
            json.dump({"nodes": [{"id": 5, "name": "OLD", "stream_url": URL, "embed_url": "e",
                                  "detail_url": "d", "status": "online"}], "edges": []}, f)
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE cameras (id INTEGER PRIMARY KEY, name TEXT, stream_url TEXT, status TEXT)")
        conn.execute("INSERT INTO cameras VALUES (5, 'OLD', ?, 'online')", (URL,))
        conn.commit()
        conn.close()
        self.patch = mock.patch.multiple(scrape, TOPOLOGY_JSON=self.topo, DB_PATH=self.db)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def result(self, **kw):
        r = {"id": 5, "name": "NEW", "detail_url": "d", "stream_url": None, "embed_url": None,
             "status": "error: timed out", "reachable": None}
        r.update(kw)
        return r

    def test_fetch_error_leaves_the_stream_url_untouched(self):
        results = [self.result()]
        report = scrape.sync_topology(results)
        scrape.sync_database(results, report["removed"])
        with open(self.topo, encoding="utf-8") as f:
            node = json.load(f)["nodes"][0]
        self.assertEqual((node["stream_url"], node["embed_url"], node["status"]), (URL, "e", "online"))
        row = sqlite3.connect(self.db).execute("SELECT stream_url, status FROM cameras WHERE id = 5").fetchone()
        self.assertEqual(row, (URL, "online"))
        self.assertEqual(report["removed"], [], "a camera whose page failed is still on the site")

    def test_page_without_a_stream_still_marks_the_camera_offline(self):
        results = [self.result(status="offline")]
        scrape.sync_topology(results)
        scrape.sync_database(results, [])
        with open(self.topo, encoding="utf-8") as f:
            node = json.load(f)["nodes"][0]
        self.assertEqual((node["stream_url"], node["status"]), (None, "offline"))
        row = sqlite3.connect(self.db).execute("SELECT stream_url, status FROM cameras WHERE id = 5").fetchone()
        self.assertEqual(row, ("", "offline"))


if __name__ == "__main__":
    unittest.main()
