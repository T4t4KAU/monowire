"""The H3 entry point must reject incompatible runtimes before worker creation."""

import os
import unittest
from unittest.mock import patch

from monowire import create_continuous_engine


class RuntimeValidationTests(unittest.TestCase):
    def test_plugin_flag_is_set_before_upstream_registration(self):
        def register():
            self.assertEqual(os.environ.get("MONOWIRE_H3_PP"), "1")
            raise RuntimeError("registration reached")

        for factory in (create_continuous_engine,):
            with (
                self.subTest(factory=factory.__name__),
                patch.dict("os.environ", {}, clear=True),
                patch("monowire.models.h3.validation.version", return_value="0.32.0+cu129"),
                patch("monowire.models.h3.plugin.register", side_effect=register),
                self.assertRaisesRegex(RuntimeError, "registration reached"),
            ):
                factory("unused", pp_size=1)

    def test_entry_points_validate_before_importing_workers(self):
        for factory in (create_continuous_engine,):
            with self.subTest(factory=factory.__name__), patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "pp_size"):
                    factory("unused", pp_size=0)
                with patch("monowire.models.h3.validation.version", return_value="0.31.0"):
                    with self.assertRaisesRegex(RuntimeError, "Expected the pinned"):
                        factory("unused", pp_size=1)
