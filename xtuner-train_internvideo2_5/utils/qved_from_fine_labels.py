import json
import os
import random
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not installed. Using default duration of 30.0s.")
    print("Install with: pip install opencv-python")

# Paths
BASE_DIR = Path('dataset')
FINE_LABELS_JSON = BASE_DIR / 'ground_truth.json'
MANIFEST_JSON = BASE_DIR / 'manifest.json'
# Output JSONL files for InternVideo2.5 training
OUTPUT_TRAIN_JSONL = Path('data/annotaions/qevd_fit_300k_train.jsonl')
OUTPUT_VAL_JSONL = Path('data/annotaions/qevd_fit_300k_val.jsonl')
OUTPUT_TEST_JSONL = Path('data/annotaions/qevd_fit_300k_test.jsonl')
USER_PROMPT_TEMPLATE = 'Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?'

# Dataset split ratios (adjustable)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42  # For reproducibility


def get_video_duration(video_path):
    """Get video duration in seconds using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        float: Video duration in seconds, or 30.0 if unable to read
    """
    if not HAS_CV2:
        return 30.0

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}, using default duration 30.0s")
            return 30.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            duration = frame_count / fps
            return round(duration, 2)
        else:
            print(f"Warning: Invalid FPS for {video_path}, using default duration 30.0s")
            return 30.0
    except Exception as e:
        print(f"Warning: Error reading video {video_path}: {e}, using default duration 30.0s")
        return 30.0


def main():
    # Load manifest to map video filenames to full paths
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    # Create reverse lookup: filename -> full_path
    filename_to_path = {}
    filename_to_exercise = {}
    for full_path, exercise in manifest.items():
        filename = os.path.basename(full_path)
        filename_to_path[filename] = full_path
        # Handle both string and dict values in manifest
        if isinstance(exercise, str):
            filename_to_exercise[filename] = exercise.replace('_', ' ')
        elif isinstance(exercise, dict):
            # For dict entries (augmented videos), extract exercise from path
            exercise_name = full_path.split(
                '/')[0] if '/' in full_path else 'unknown'
            filename_to_exercise[filename] = exercise_name.replace('_', ' ')
        else:
            filename_to_exercise[filename] = str(exercise).replace('_', ' ')

    # Load fine-grained labels
    with open(FINE_LABELS_JSON) as f:
        fine_labels = json.load(f)

    # Convert to Mobile-VideoGPT format
    output_data = []

    for record in fine_labels:
        video_path = record.get('video_path', '')
        # Extract filename from path (handles ./ prefix)
        filename = os.path.basename(video_path)

        # Look up full path in manifest
        if filename not in filename_to_path:
            print(f'Warning: {filename} not found in manifest, skipping')
            continue

        full_video_path = filename_to_path[filename]
        exercise = filename_to_exercise[filename]

        # Remove 'dataset/' prefix if present to make path relative to data_path
        # data_path is 'dataset', so videos should be 'exercise_name/video.mp4'
        if full_video_path.startswith('dataset/'):
            relative_video_path = full_video_path[len('dataset/'):]
        else:
            relative_video_path = full_video_path

        # Get assistant answer from most descriptive label
        if 'labels_descriptive' in record and record['labels_descriptive']:
            assistant_answer = record['labels_descriptive']
        elif 'labels' in record and record['labels']:
            assistant_answer = record['labels'][0] if isinstance(
                record['labels'], list) else record['labels']
        else:
            assistant_answer = 'No feedback available.'

        # Ensure assistant answer is a single string
        if isinstance(assistant_answer, list):
            assistant_answer = '\n'.join(
                str(item) for item in assistant_answer)
        else:
            assistant_answer = str(assistant_answer)

        user_prompt = USER_PROMPT_TEMPLATE  # No longer using exercise name in prompt

        # Get actual video duration
        full_video_file_path = BASE_DIR / relative_video_path
        video_duration = get_video_duration(full_video_file_path)

        output_data.append({
            'video':
            relative_video_path,
            'conversations': [{
                'from': 'human',
                'value': user_prompt
            }, {
                'from': 'gpt',
                'value': assistant_answer
            }],
            'duration': video_duration,
            'split':
            'train'  # Will be updated during split
        })

    # Shuffle data for random split
    random.seed(RANDOM_SEED)
    random.shuffle(output_data)

    # Calculate split indices
    total_count = len(output_data)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)

    # Split the data
    train_data = output_data[:train_end]
    val_data = output_data[train_end:val_end]
    test_data = output_data[val_end:]

    # Update split labels
    for item in train_data:
        item['split'] = 'train'
    for item in val_data:
        item['split'] = 'val'
    for item in test_data:
        item['split'] = 'test'

    # Write output JSONLs (one JSON object per line)
    OUTPUT_TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # Remove split field, not needed in JSONL (duration already set)
    for item in train_data:
        del item['split']

    for item in val_data:
        del item['split']

    for item in test_data:
        del item['split']

    with open(OUTPUT_TRAIN_JSONL, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(OUTPUT_VAL_JSONL, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    with open(OUTPUT_TEST_JSONL, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n{'='*60}")
    print(f'Dataset Split Summary')
    print(f"{'='*60}")
    print(f'Total videos: {total_count}')
    print(f'Exercise classes: {len(set(filename_to_exercise.values()))}')
    print(f'\nSplit Distribution:')
    print(
        f'  Train: {len(train_data)} samples ({len(train_data)/total_count*100:.1f}%)'
    )
    print(
        f'  Val:   {len(val_data)} samples ({len(val_data)/total_count*100:.1f}%)'
    )
    print(
        f'  Test:  {len(test_data)} samples ({len(test_data)/total_count*100:.1f}%)'
    )
    print(f'\nOutput files (JSONL format for InternVideo2.5):')
    print(f'  Train: {OUTPUT_TRAIN_JSONL}')
    print(f'  Val:   {OUTPUT_VAL_JSONL}')
    print(f'  Test:  {OUTPUT_TEST_JSONL}')
    print(f"{'='*60}")
    if HAS_CV2:
        print(f'\nNote: Video durations extracted dynamically from video files.')
    else:
        print(f'\nNote: opencv-python not available, using default duration of 30.0s.')
        print(f'For accurate durations, install: pip install opencv-python')


if __name__ == '__main__':
    main()
