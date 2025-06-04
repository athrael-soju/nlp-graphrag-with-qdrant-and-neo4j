import unittest
from unittest.mock import MagicMock, patch, call
import os
import tempfile
import shutil

# Direct import, assuming tests are run from project root or PYTHONPATH is set.
from graphrag.core.video_ingest import (
    extract_frames_into_segments,
    extract_video_metadata,
    process_video_for_visual_analysis,
    VideoSegment
)
import cv2 # Import cv2 for CAP_PROP_* constants, moved after main module import

class TestVideoIngest(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = tempfile.mkdtemp(prefix="graphrag_test_video_ingest_")
        self.mock_video_path = os.path.join(self.test_output_dir, "test_video.mp4")
        with open(self.mock_video_path, "w") as f:
            f.write("dummy video content")

    def tearDown(self):
        shutil.rmtree(self.test_output_dir)

    @patch('graphrag.core.video_ingest.cv2.VideoCapture')
    @patch('graphrag.core.video_ingest.cv2.imwrite')
    @patch('graphrag.core.video_ingest.os.makedirs')
    @patch('graphrag.core.video_ingest.os.path.exists')
    def test_extract_frames_into_segments_typical_case(
        self, mock_path_exists, mock_os_makedirs, mock_imwrite, MockVideoCapture
    ):
        mock_path_exists.return_value = True

        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = True

        # Correctly use cv2 constants
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300.0 # 10 seconds video
        }.get(prop, 0.0)

        # Simulate reading 300 frames
        # Create a list of (True, MagicMock()) tuples for read() side_effect
        mock_frames = [(True, MagicMock(name=f"frame_{i}")) for i in range(300)]
        mock_cap_instance.read.side_effect = mock_frames + [(False, None)]


        segments = extract_frames_into_segments(
            video_path=self.mock_video_path,
            output_dir=self.test_output_dir,
            frames_per_second_to_extract=1,
            segment_duration_seconds=3
        )

        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0]['segment_id'], 0)
        self.assertEqual(len(segments[0]['frame_paths']), 3)
        self.assertEqual(segments[3]['segment_id'], 3)
        self.assertEqual(len(segments[3]['frame_paths']), 1)
        self.assertEqual(mock_imwrite.call_count, 10)

        # Check one specific call to imwrite
        # The frame data passed to imwrite should be one of the MagicMock objects from mock_frames
        expected_frame_path_0 = os.path.join(self.test_output_dir, "frame_000000.jpg")
        # The actual frame object written is mock_frames[0][1] because frame_skip_interval will be 30 (30fps / 1fps_extract)
        # so the first saved frame is current_frame_number=0 (mock_frames[0][1])
        # second saved frame is current_frame_number=30 (mock_frames[30][1])
        self.assertTrue(any(
            call_args[0][0] == expected_frame_path_0 and call_args[0][1] == mock_frames[0][1]
            for call_args in mock_imwrite.call_args_list
        ))


    @patch('graphrag.core.video_ingest.cv2.VideoCapture')
    @patch('graphrag.core.video_ingest.os.path.exists')
    def test_extract_video_metadata_success(self, mock_path_exists, MockVideoCapture):
        mock_path_exists.return_value = True
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 250.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0
        }.get(prop, 0.0)

        metadata = extract_video_metadata(self.mock_video_path)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['video_title'], "test_video.mp4")
        self.assertEqual(metadata['duration_seconds'], 10.0)
        self.assertEqual(metadata['resolution'], "1920x1080")

    @patch('graphrag.core.video_ingest.cv2.VideoCapture')
    @patch('graphrag.core.video_ingest.os.path.exists')
    def test_extract_video_metadata_open_fail(self, mock_path_exists, MockVideoCapture):
        mock_path_exists.return_value = True
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = False
        metadata = extract_video_metadata(self.mock_video_path)
        self.assertIsNone(metadata)

    @patch('graphrag.core.video_ingest.extract_video_metadata')
    @patch('graphrag.core.video_ingest.extract_frames_into_segments')
    @patch('graphrag.core.video_ingest.os.path.exists')
    @patch('graphrag.core.video_ingest.os.makedirs')
    def test_process_video_for_visual_analysis_success(
        self, mock_os_makedirs, mock_os_path_exists, mock_extract_frames, mock_extract_metadata
    ):
        video_filename_stem = os.path.splitext(os.path.basename(self.mock_video_path))[0]
        expected_frame_output_dir = os.path.join(self.test_output_dir, video_filename_stem + "_frames")

        def path_exists_side_effect(path):
            if path == self.mock_video_path: return True
            if path == expected_frame_output_dir: return False # Ensure makedirs is called
            return True # Default to True for other os.path.exists checks if any
        mock_os_path_exists.side_effect = path_exists_side_effect

        dummy_metadata = {"video_title": "test_video.mp4", "duration_seconds": 10.0}
        dummy_segments = [ VideoSegment(segment_id=0, start_time=0.0, end_time=5.0, frame_paths=["f1.jpg"]) ]
        mock_extract_metadata.return_value = dummy_metadata
        mock_extract_frames.return_value = dummy_segments

        result = process_video_for_visual_analysis(
            video_path=self.mock_video_path,
            base_output_dir=self.test_output_dir
        )

        self.assertIsNotNone(result)
        self.assertEqual(result['video_metadata'], dummy_metadata)
        self.assertEqual(result['segments'], dummy_segments)
        mock_os_makedirs.assert_called_once_with(expected_frame_output_dir)
        mock_extract_metadata.assert_called_once_with(self.mock_video_path)
        mock_extract_frames.assert_called_once_with(
            video_path=self.mock_video_path,
            output_dir=expected_frame_output_dir,
            frames_per_second_to_extract=1, # Default value
            segment_duration_seconds=5    # Default value
        )

if __name__ == '__main__':
    unittest.main()
