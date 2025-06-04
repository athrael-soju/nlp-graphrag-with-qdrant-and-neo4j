import cv2
import os
import json
import logging
import tempfile
from typing import Dict, List, TypedDict, Optional, Any

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoSegment(TypedDict):
    segment_id: int
    start_time: float
    end_time: float
    frame_paths: List[str]
    # Potentially add other segment-level metadata here later

def extract_frames_into_segments(
    video_path: str,
    output_dir: str,
    frames_per_second_to_extract: int = 1,
    segment_duration_seconds: int = 5
) -> List[VideoSegment]:
    """
    Extracts frames from a video file and groups them into temporal segments.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        frames_per_second_to_extract (int): Number of frames to extract per second of video.
        segment_duration_seconds (int): Duration of each video segment in seconds.

    Returns:
        List[VideoSegment]: A list of video segments, each containing paths to its frames.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        logging.warning(f"Could not read FPS from video: {video_path}. Assuming 30 FPS for frame skipping logic.")
        video_fps = 30 # Default if FPS is not available

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_video_frames / video_fps if video_fps > 0 else 0

    logging.info(f"Video properties: FPS={video_fps:.2f}, Total Frames={total_video_frames}, Duration={video_duration_seconds:.2f}s")

    segments: List[VideoSegment] = []
    current_frame_number = 0
    frame_save_count = 0

    frame_skip_interval = int(video_fps / frames_per_second_to_extract) if frames_per_second_to_extract > 0 and video_fps >= frames_per_second_to_extract else 1


    current_segment_id = 0
    current_segment_start_time = 0.0
    current_segment_frames: List[str] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = current_frame_number / video_fps

        if current_time_sec >= current_segment_start_time + segment_duration_seconds:
            if current_segment_frames:
                segments.append(VideoSegment(
                    segment_id=current_segment_id,
                    start_time=current_segment_start_time,
                    end_time=current_segment_start_time + segment_duration_seconds,
                    frame_paths=list(current_segment_frames)
                ))
                logging.info(f"Finalized segment {current_segment_id} with {len(current_segment_frames)} frames from {current_segment_start_time:.2f}s to {current_segment_start_time + segment_duration_seconds:.2f}s.")
                current_segment_frames.clear()
            current_segment_id += 1
            current_segment_start_time = current_segment_id * segment_duration_seconds

        if current_frame_number % frame_skip_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_save_count:06d}.jpg")
            try:
                cv2.imwrite(frame_filename, frame)
                current_segment_frames.append(frame_filename)
                frame_save_count +=1
            except Exception as e:
                logging.error(f"Error writing frame {frame_filename}: {e}")

        current_frame_number += 1

    if current_segment_frames:
        actual_end_time = current_time_sec
        segments.append(VideoSegment(
            segment_id=current_segment_id,
            start_time=current_segment_start_time,
            end_time=actual_end_time,
            frame_paths=list(current_segment_frames)
        ))
        logging.info(f"Finalized last segment {current_segment_id} with {len(current_segment_frames)} frames from {current_segment_start_time:.2f}s to {actual_end_time:.2f}s.")

    cap.release()
    logging.info(f"Extracted {frame_save_count} frames in total, grouped into {len(segments)} segments.")
    return segments


def extract_video_metadata(video_path: str) -> Optional[Dict]:
    if not os.path.exists(video_path):
        logging.error(f"Metadata extraction: Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Metadata extraction: Error opening video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = 0
    if fps > 0 and total_frames > 0:
        duration_seconds = total_frames / fps

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution = f"{int(width)}x{int(height)}" if width > 0 and height > 0 else "unknown"
    cap.release()

    metadata = {
        "video_title": os.path.basename(video_path),
        "duration_seconds": duration_seconds,
        "resolution": resolution,
        "fps": fps,
        "total_frames": total_frames,
        "platform": "unknown_platform",
        "creator": "unknown_creator",
        "upload_date": "YYYY-MM-DD",
        "engagement_metrics": {
            "views": 0, "likes": 0, "comments": 0
        }
    }
    logging.info(f"Extracted metadata for video {video_path}: {metadata}")
    return metadata


def process_video_for_visual_analysis(
    video_path: str,
    base_output_dir: str,
    frames_per_second_to_extract: int = 1,
    segment_duration_seconds: int = 5
) -> Optional[Dict[str, Any]]:
    logging.info(f"Starting processing for video: {video_path}")
    if not os.path.exists(video_path):
        logging.error(f"Video not found for processing: {video_path}")
        return None

    video_meta = extract_video_metadata(video_path)
    if not video_meta:
        logging.error(f"Failed to extract metadata for {video_path}. Aborting.")
        return None

    video_filename_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_frame_output_dir = os.path.join(base_output_dir, video_filename_stem + "_frames")
    if not os.path.exists(video_frame_output_dir):
        os.makedirs(video_frame_output_dir)
    logging.info(f"Frames will be stored in: {video_frame_output_dir}")

    segments = extract_frames_into_segments(
        video_path=video_path,
        output_dir=video_frame_output_dir,
        frames_per_second_to_extract=frames_per_second_to_extract,
        segment_duration_seconds=segment_duration_seconds
    )

    logging.info(f"Successfully processed video {video_path}. Metadata and {len(segments)} segments extracted.")
    return {
        "video_metadata": video_meta,
        "segments": segments
    }


if __name__ == '__main__':
    import numpy as np
    example_base_output_dir = tempfile.mkdtemp(prefix="graphrag_video_processing_example_")
    logging.info(f"Example output base directory: {example_base_output_dir}")
    sample_video_path = "sample.mp4"
    dummy_video_duration_seconds = 12
    dummy_fps = 10
    dummy_total_frames = dummy_video_duration_seconds * dummy_fps
    full_sample_video_path = os.path.join(example_base_output_dir, sample_video_path)

    if not os.path.exists(full_sample_video_path):
        logging.info(f"Creating a dummy video file: {full_sample_video_path}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(full_sample_video_path, fourcc, float(dummy_fps), (320, 240))
            if out.isOpened():
                for i in range(dummy_total_frames):
                    frame_np = np.zeros((240, 320, 3), dtype=np.uint8)
                    frame_np[:, :] = ((i * 2) % 255, (i * 3) % 255, (i * 5) % 255)
                    out.write(frame_np)
                out.release()
                logging.info(f"Dummy video created: {full_sample_video_path}")
            else:
                logging.error(f"Failed to open VideoWriter for dummy video {full_sample_video_path}.")
        except Exception as e:
            logging.error(f"Error creating dummy video: {e}. Creating empty file as fallback.")
            with open(full_sample_video_path, 'w') as f: f.write('')

    if os.path.exists(full_sample_video_path) and os.path.getsize(full_sample_video_path) > 0:
        processing_result = process_video_for_visual_analysis(
            video_path=full_sample_video_path,
            base_output_dir=example_base_output_dir,
            frames_per_second_to_extract=1,
            segment_duration_seconds=5
        )
        if processing_result:
            logging.info("\n--- Video Processing Result ---")
            logging.info(f"Overall Video Metadata: {json.dumps(processing_result['video_metadata'], indent=2)}")
            logging.info(f"Number of Segments: {len(processing_result['segments'])}")
            for i, segment in enumerate(processing_result['segments']):
                logging.info(f"\n--- Details for Segment {segment['segment_id']} ---")
                logging.info(f"  Start Time: {segment['start_time']:.2f}s, End Time: {segment['end_time']:.2f}s")
                logging.info(f"  Number of frames in this segment: {len(segment['frame_paths'])}")
                if segment['frame_paths']:
                    logging.info(f"  First frame path in segment: {segment['frame_paths'][0]}")
            logging.info("\n--- End of Segment Details ---")
        else:
            logging.error("Video processing failed.")
    else:
        logging.error(f"Dummy sample video {full_sample_video_path} not found or is empty. Skipping processing example.")

    logging.info(f"\nExample finished. All outputs (dummy video, frames) are in: {example_base_output_dir}")
    logging.info("Consider deleting this directory manually if it's not in a system temp location that gets auto-cleaned.")
