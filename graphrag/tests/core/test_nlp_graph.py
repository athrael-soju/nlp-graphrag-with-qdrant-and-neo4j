import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Optional

# Attempt to import the target module
try:
    from graphrag.core.nlp_graph import NLPGraphBuilder
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
    from graphrag.core.nlp_graph import NLPGraphBuilder

# Mock NLTK related imports if NLTK data isn't downloaded in test environment
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words('english'))
    except Exception:
        STOPWORDS = set()
        if 'graphrag.core.nlp_graph' in sys.modules:
            sys.modules['graphrag.core.nlp_graph'].STOPWORDS = STOPWORDS


class MockNeo4jConnection:
    def __init__(self):
        self.queries_run = []
        self.closed = False

    def run_query(self, query: str, params: Optional[dict] = None) -> MagicMock:
        self.queries_run.append({"query": query.strip(), "params": params})
        return MagicMock()

    def close(self):
        self.closed = True

    def clear_queries(self):
        self.queries_run = []


class TestNLPGraphVideoSegments(unittest.TestCase):

    def setUp(self):
        self.mock_neo4j_conn = MockNeo4jConnection()
        self.graph_builder = NLPGraphBuilder(neo4j_conn=self.mock_neo4j_conn)
        self.sample_video_id = "test_video_001"
        self.sample_segments_data = [
            {"id": "segment_0", "sequence_number": 0, "start_time": 0.0, "end_time": 5.0, "frame_paths": ["f1.jpg"]},
            {"id": "segment_1", "sequence_number": 1, "start_time": 5.0, "end_time": 10.0, "frame_paths": ["f2.jpg", "f3.jpg"]},
            {"id": "segment_2", "sequence_number": 2, "start_time": 10.0, "end_time": 12.5, "frame_paths": ["f4.jpg"]},
        ]

    def test_store_video_segments_creates_video_node(self):
        self.graph_builder.store_video_segments_in_neo4j(self.sample_video_id, self.sample_segments_data)
        found_video_merge = False
        for item in self.mock_neo4j_conn.queries_run:
            if "MERGE (v:Video {id: $video_id})" in item["query"] and \
               item["params"]["video_id"] == self.sample_video_id:
                found_video_merge = True
                break
        self.assertTrue(found_video_merge, "Query to MERGE Video node not found or incorrect.")

    def test_store_video_segments_creates_segment_nodes_and_has_segment_rels(self):
        self.graph_builder.store_video_segments_in_neo4j(self.sample_video_id, self.sample_segments_data)
        video_segment_merges = 0
        has_segment_rels = 0
        expected_node_props = [
            {'id': 'segment_0', 'video_id': self.sample_video_id, 'start_time': 0.0, 'end_time': 5.0, 'sequence_number': 0},
            {'id': 'segment_1', 'video_id': self.sample_video_id, 'start_time': 5.0, 'end_time': 10.0, 'sequence_number': 1},
            {'id': 'segment_2', 'video_id': self.sample_video_id, 'start_time': 10.0, 'end_time': 12.5, 'sequence_number': 2},
        ]
        for i, segment_data in enumerate(self.sample_segments_data):
            segment_id = segment_data["id"]
            found_segment_merge = False
            found_has_segment_rel = False
            for item in self.mock_neo4j_conn.queries_run:
                query = item["query"]
                params = item["params"]
                if "MERGE (s:VideoSegment {id: $segment_id})" in query and params.get("segment_id") == segment_id:
                    # Check if all expected properties are in the query parameters
                    self.assertEqual(params["props"], expected_node_props[i])
                    video_segment_merges +=1
                    found_segment_merge = True
                if "MERGE (vid)-[:HAS_SEGMENT]->(s)" in query and params.get("segment_id") == segment_id:
                    has_segment_rels +=1
                    found_has_segment_rel = True
            self.assertTrue(found_segment_merge, f"Query to MERGE VideoSegment node {segment_id} not found.")
            self.assertTrue(found_has_segment_rel, f"Query to link Video to VideoSegment {segment_id} not found.")
        self.assertEqual(video_segment_merges, len(self.sample_segments_data))
        self.assertEqual(has_segment_rels, len(self.sample_segments_data))

    def test_store_video_segments_creates_before_after_relationships(self):
        self.graph_builder.store_video_segments_in_neo4j(self.sample_video_id, self.sample_segments_data)
        before_after_rels_count = 0
        expected_pairs = [("segment_0", "segment_1"), ("segment_1", "segment_2")]
        for item in self.mock_neo4j_conn.queries_run:
            query = item["query"]
            params = item["params"]
            if "MERGE (s1)-[:BEFORE]->(s2)" in query and "MERGE (s2)-[:AFTER]->(s1)" in query:
                pair_found = False
                for prev_id, curr_id in expected_pairs:
                    if params.get("prev_id") == prev_id and params.get("curr_id") == curr_id:
                        before_after_rels_count +=1
                        pair_found = True
                        break
                self.assertTrue(pair_found, f"Unexpected BEFORE/AFTER pair: {params}")
        self.assertEqual(before_after_rels_count, len(self.sample_segments_data) - 1)

    def test_store_video_segments_empty_list(self):
        self.mock_neo4j_conn.clear_queries()
        self.graph_builder.store_video_segments_in_neo4j(self.sample_video_id, [])
        # Expect 1 query for the Video node merge, as it happens before the empty check.
        self.assertEqual(len(self.mock_neo4j_conn.queries_run), 1)
        self.assertIn("MERGE (v:Video {id: $video_id})", self.mock_neo4j_conn.queries_run[0]["query"])

    def test_store_video_segments_segment_missing_id(self):
        segments_with_missing_id = [
            {"id": "segment_A", "sequence_number": 0, "start_time": 0.0, "end_time": 5.0},
            {"sequence_number": 1, "start_time": 5.0, "end_time": 10.0},
            {"id": "segment_C", "sequence_number": 2, "start_time": 10.0, "end_time": 15.0},
        ]
        self.mock_neo4j_conn.clear_queries()
        self.graph_builder.store_video_segments_in_neo4j(self.sample_video_id, segments_with_missing_id)
        created_segment_ids = set()
        for item in self.mock_neo4j_conn.queries_run:
            if "MERGE (s:VideoSegment {id: $segment_id})" in item["query"]:
                created_segment_ids.add(item["params"]["segment_id"])
        self.assertIn("segment_A", created_segment_ids)
        self.assertIn("segment_C", created_segment_ids)
        self.assertEqual(len(created_segment_ids), 2, "Should only create segments with IDs.")

        before_after_rel_queries = [
            item for item in self.mock_neo4j_conn.queries_run
            if "MERGE (s1)-[:BEFORE]->(s2)" in item["query"]
        ]
        # Since segment with missing ID is skipped, no consecutive pair (A,C) is formed directly by loop.
        self.assertEqual(len(before_after_rel_queries), 0)

if __name__ == '__main__':
    unittest.main()
