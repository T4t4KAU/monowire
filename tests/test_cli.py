"""Image input validation and request construction without model allocation."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from monowire.cli import main


class ImageCLITests(unittest.TestCase):
    def invoke(self, *options):
        with patch(
            "sys.argv", ["monowire", "--model", "unused", "--output", "unused.pt", *options]
        ):
            main()

    def test_invalid_image_options_fail_before_worker_creation(self):
        cases = (
            ("--prompt", "scene", "--task", "fl2va"),
            ("--prompt", "scene", "--task", "t2va", "--image", "unused.png"),
            ("--prompt", "one", "--prompt", "two", "--image", "unused.png"),
            ("--prompt", "scene", "--image", "/missing/monowire-image.png"),
        )
        for options in cases:
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                patch("monowire.models.h3.runtime.create_continuous_engine") as create,
                self.assertRaises(SystemExit) as caught,
            ):
                self.invoke(*options)
            self.assertEqual(caught.exception.code, 2)
            create.assert_not_called()

    def test_images_are_rgb_and_paired_with_their_prompts(self):
        engine = Mock(is_ingress=True, ingress_replicas=1)
        engine.generate.return_value = []
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder, name) for name in ("red.png", "gray.png")]
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths[0])
            Image.new("L", (8, 8), 37).save(paths[1])
            with (
                patch("monowire.models.h3.runtime.create_continuous_engine", return_value=engine),
                patch.dict(
                    "sys.modules",
                    {
                        "vllm_omni.inputs.data": SimpleNamespace(
                            OmniDiffusionSamplingParams=SimpleNamespace
                        )
                    },
                ),
                patch.object(torch, "save"),
            ):
                self.invoke(
                    "--prompt",
                    "red",
                    "--prompt",
                    "gray",
                    "--image",
                    str(paths[0]),
                    "--image",
                    str(paths[1]),
                )
        prompts, sampling = engine.generate.call_args.args
        self.assertEqual(sampling.extra_args["task"], "fl2va")
        self.assertEqual([p["prompt"] for p in prompts], ["red", "gray"])
        images = [p["multi_modal_data"]["image"] for p in prompts]
        self.assertEqual([im.mode for im in images], ["RGB", "RGB"])
        self.assertEqual([im.getpixel((0, 0)) for im in images], [(255, 0, 0), (37, 37, 37)])
        engine.close.assert_called_once()
