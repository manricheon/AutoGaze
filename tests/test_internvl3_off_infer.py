from repro.internvl3_off_infer import build_video_question, get_uniform_frame_indices


def test_get_uniform_frame_indices_samples_segment_centers():
    assert get_uniform_frame_indices(max_frame=99, num_segments=4) == [12, 37, 62, 87]


def test_build_video_question_prefixes_one_image_token_per_frame():
    question = build_video_question([1, 2, 1], "What happens?")

    assert question == "Frame1: <image>\nFrame2: <image>\nFrame3: <image>\nWhat happens?"
