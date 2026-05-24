import json

from scripts.convert_video_task_dataset import (
    DATASET_PRESETS,
    convert_dataset_rows,
    convert_dataset_to_manifest,
    parse_args,
)


def test_msrvtt_caption_converter_groups_references_by_video():
    rows = [
        {"video": "video0001.mp4", "caption": "a person cooks", "category": 16},
        {"video": "video0001.mp4", "caption": "someone prepares food", "category": 16},
        {"video_path": "video0002.mp4", "text": "a dog runs"},
    ]

    converted = convert_dataset_rows(rows, dataset_preset="msrvtt-caption")

    assert converted == [
        {
            "sample_id": "video0001",
            "video_path": "video0001.mp4",
            "references": ["a person cooks", "someone prepares food"],
            "source": "VLM2Vec/MSR-VTT",
            "category": "16",
        },
        {
            "sample_id": "video0002",
            "video_path": "video0002.mp4",
            "references": ["a dog runs"],
            "source": "VLM2Vec/MSR-VTT",
        },
    ]


def test_ucf101_action_converter_strips_leading_slash_and_keeps_label():
    rows = [
        {
            "clip_name": "v_Swing_g05_c02",
            "clip_path": "/train/Swing/v_Swing_g05_c02.avi",
            "label": "Swing",
        }
    ]

    converted = convert_dataset_rows(rows, dataset_preset="ucf101-action")

    assert converted == [
        {
            "sample_id": "v_Swing_g05_c02",
            "video_path": "train/Swing/v_Swing_g05_c02.avi",
            "label": "Swing",
            "source": "bitmind/UCF101-Videos",
        }
    ]


def test_nextqa_videoqa_converter_maps_choices_and_numeric_answer_to_letter():
    rows = [
        {
            "video": "123.mp4",
            "question": "Why does she wipe her eyes?",
            "answer": 1,
            "a0": "remove makeup",
            "a1": "wipe tears",
            "a2": "dance",
            "type": "CW",
        }
    ]

    converted = convert_dataset_rows(rows, dataset_preset="nextqa-videoqa")

    assert converted == [
        {
            "sample_id": "123",
            "video_path": "123.mp4",
            "question": "Why does she wipe her eyes?",
            "answer": "B",
            "choices": ["remove makeup", "wipe tears", "dance"],
            "source": "VLM2Vec/nextqa",
            "category": "CW",
        }
    ]


def test_activitynet_caption_converter_preserves_temporal_metadata():
    rows = [{"video_id": "v_abc", "caption": "a person jumps", "start_time": 1.2, "end_time": 3.4, "duration": 5.0}]

    converted = convert_dataset_rows(rows, dataset_preset="activitynet-caption")

    assert converted == [
        {
            "sample_id": "v_abc",
            "video_path": "v_abc",
            "references": ["a person jumps"],
            "source": "qingy2024/ActivityNet-Captions",
            "duration": 5.0,
            "temporal_segments": [{"start_time": 1.2, "end_time": 3.4, "caption": "a person jumps"}],
        }
    ]


def test_convert_dataset_to_manifest_writes_jsonl_from_csv(tmp_path):
    source = tmp_path / "ucf.csv"
    source.write_text("clip_name,clip_path,label\nclip1,/train/Run/clip1.avi,Run\n")
    output = tmp_path / "manifest.jsonl"

    result = convert_dataset_to_manifest(
        input_path=source,
        output_path=output,
        dataset_preset="ucf101-action",
        limit=1,
    )

    assert result["rows_written"] == 1
    assert result["task_type"] == "action_classification"
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {
            "sample_id": "clip1",
            "video_path": "train/Run/clip1.avi",
            "label": "Run",
            "source": "bitmind/UCF101-Videos",
        }
    ]


def test_converter_cli_defaults_to_msrvtt_caption_preset():
    args = parse_args(["--input", "data.csv", "--output", "manifest.jsonl"])

    assert args.dataset_preset == "msrvtt-caption"
    assert DATASET_PRESETS["msrvtt-caption"]["task_type"] == "captioning"
