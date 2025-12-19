from pathlib import Path

import pytest

from scope.cli.render_timeline import (
    TimelineFile,
    _build_linear_timestep_schedule,
    _parse_denoising_steps,
    _validate_resolution,
)


def test_parse_denoising_steps():
    assert _parse_denoising_steps("1000, 750,500") == [1000, 750, 500]


def test_build_linear_timestep_schedule_matches_krea_default():
    assert _build_linear_timestep_schedule(4, start=1000, end=250) == [
        1000,
        750,
        500,
        250,
    ]
    assert _build_linear_timestep_schedule(6, start=1000, end=250) == [
        1000,
        850,
        700,
        550,
        400,
        250,
    ]


def test_validate_resolution_requires_multiple_of_16():
    _validate_resolution(320, 576)
    with pytest.raises(ValueError):
        _validate_resolution(321, 576)


def test_timeline_model_parses_repo_examples():
    krea_timeline = TimelineFile.model_validate(
        {
            "prompts": [
                {
                    "startTime": 0,
                    "endTime": 1.0,
                    "transitionSteps": 4,
                    "temporalInterpolationMethod": "slerp",
                    "prompts": [{"text": "ANCHOR: a cat in a hat", "weight": 100}],
                }
            ],
            "settings": {
                "pipelineId": "krea-realtime-video",
                "inputMode": "text",
                "resolution": {"height": 320, "width": 576},
                "seed": 42,
                "denoisingSteps": [1000, 750, 500, 250],
                "manageCache": True,
                "quantization": None,
                "kvCacheAttentionBias": 0.3,
                "loras": [],
                "loraMergeStrategy": "permanent_merge",
            },
            "version": "2.1",
        }
    )
    assert krea_timeline.settings.pipelineId == "krea-realtime-video"
    assert krea_timeline.prompts
    assert krea_timeline.prompts[0].prompt_items()

    example_timeline = TimelineFile.model_validate_json(
        Path(
            "src/scope/core/pipelines/streamdiffusionv2/docs/examples/timeline-evolution.json"
        ).read_text()
    )
    assert example_timeline.settings.pipelineId == "streamdiffusionv2"
    assert example_timeline.prompts
    assert example_timeline.prompts[0].prompt_items()
