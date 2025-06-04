import unittest
import os # Ensure os is imported
from typing import List, Dict, Any, Optional

# Attempt to import the target module
try:
    from graphrag.core.visual_analyzer import (
        analyze_with_rf_detr,
        describe_scene_with_blip2,
        analyze_with_gemini_vision,
        create_unified_visual_analysis
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
    from graphrag.core.visual_analyzer import (
        analyze_with_rf_detr,
        describe_scene_with_blip2,
        analyze_with_gemini_vision,
        create_unified_visual_analysis
    )

class TestVisualAnalyzer(unittest.TestCase):

    def setUp(self):
        self.dummy_frames = [
            "dummy_frame_1.jpg",
            "dummy_frame_2.jpg",
            "dummy_frame_3.jpg"
        ]

    def test_analyze_with_rf_detr_placeholder(self):
        results = analyze_with_rf_detr(self.dummy_frames)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.dummy_frames))
        for i, item in enumerate(results):
            self.assertIsInstance(item, dict)
            self.assertEqual(item["frame_path"], self.dummy_frames[i])
            self.assertIn("objects", item)
            self.assertIsInstance(item["objects"], list)

    def test_describe_scene_with_blip2_placeholder(self):
        descriptions = describe_scene_with_blip2(self.dummy_frames)
        self.assertIsInstance(descriptions, list)
        self.assertEqual(len(descriptions), len(self.dummy_frames))
        for i, desc in enumerate(descriptions):
            self.assertIsInstance(desc, str)
            self.assertIn(os.path.basename(self.dummy_frames[i]), desc)

    def test_analyze_with_gemini_vision_placeholder(self):
        results = analyze_with_gemini_vision(self.dummy_frames)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.dummy_frames))
        for i, item in enumerate(results):
            self.assertIsInstance(item, dict)
            self.assertEqual(item["frame_path"], self.dummy_frames[i])
            self.assertIn("description", item)
            self.assertIn("tags", item)
            self.assertIsInstance(item["tags"], list)

    def test_create_unified_visual_analysis_all_inputs(self):
        mock_rf_results = analyze_with_rf_detr(self.dummy_frames)
        mock_blip_descs = describe_scene_with_blip2(self.dummy_frames)
        mock_gemini_results = analyze_with_gemini_vision(self.dummy_frames)

        unified_results = create_unified_visual_analysis(
            image_frames=self.dummy_frames,
            rf_detr_results=mock_rf_results,
            blip2_descriptions=mock_blip_descs,
            gemini_results=mock_gemini_results
        )
        self.assertEqual(len(unified_results), len(self.dummy_frames))
        for i, item in enumerate(unified_results):
            self.assertEqual(item["frame_path"], self.dummy_frames[i])
            self.assertEqual(item["source_video_frame_number"], i)
            self.assertIn("rf_detr_analysis", item)
            self.assertIn("blip2_description", item)
            self.assertIn("gemini_analysis", item)
            self.assertEqual(item["rf_detr_analysis"], mock_rf_results[i]["objects"])
            self.assertEqual(item["blip2_description"], mock_blip_descs[i])
            expected_gemini_data = mock_gemini_results[i].copy()
            expected_gemini_data.pop("frame_path", None)
            self.assertEqual(item["gemini_analysis"], expected_gemini_data)

    def test_create_unified_visual_analysis_some_inputs_missing(self):
        mock_rf_results = analyze_with_rf_detr(self.dummy_frames)
        unified_results = create_unified_visual_analysis(
            image_frames=self.dummy_frames,
            rf_detr_results=mock_rf_results,
            blip2_descriptions=None,
            gemini_results=None
        )
        self.assertEqual(len(unified_results), len(self.dummy_frames))
        for i, item in enumerate(unified_results):
            self.assertEqual(item["frame_path"], self.dummy_frames[i])
            self.assertIn("rf_detr_analysis", item)
            self.assertIsNone(item["blip2_description"])
            self.assertIsNone(item["gemini_analysis"])
            self.assertEqual(item["rf_detr_analysis"], mock_rf_results[i]["objects"])

    def test_create_unified_visual_analysis_mismatched_lengths_optional_inputs(self):
        mock_rf_results = analyze_with_rf_detr(self.dummy_frames[:1]) # Only for first frame
        unified_results = create_unified_visual_analysis(
            image_frames=self.dummy_frames,
            rf_detr_results=mock_rf_results,
            blip2_descriptions=[],
            gemini_results=None
        )
        self.assertEqual(len(unified_results), len(self.dummy_frames))
        self.assertIsNotNone(unified_results[0]["rf_detr_analysis"])
        self.assertIsNone(unified_results[0]["blip2_description"])
        self.assertIsNone(unified_results[0]["gemini_analysis"])
        for i in range(1, len(self.dummy_frames)):
            self.assertIsNone(unified_results[i]["rf_detr_analysis"])
            self.assertIsNone(unified_results[i]["blip2_description"])
            self.assertIsNone(unified_results[i]["gemini_analysis"])

    def test_create_unified_visual_analysis_empty_frames(self):
        unified_results = create_unified_visual_analysis(image_frames=[])
        self.assertEqual(len(unified_results), 0)

if __name__ == '__main__':
    unittest.main()
