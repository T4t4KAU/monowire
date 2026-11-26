"""Exercise the native split-forward methods without loading model weights."""

import ast
import math
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


class H3OutputContextTests(unittest.TestCase):
    def test_context_does_not_own_input_and_preserves_output_masks(self):
        path = Path(__file__).resolve().parents[1] / (
            "third_party/vllm-omni/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"
        )
        tree = ast.parse(path.read_text())
        model_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3DiTModel"
        )
        methods = [
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"prepare_transformer_forward", "finish_transformer_forward"}
        ]
        inputs = dict(
            x=torch.zeros(1, 3, 2),
            audio_x=torch.zeros(1, 1, 2),
            img_position_ids=torch.zeros(3, 3),
            unique_timesteps=torch.zeros(1),
            inverse_indices=torch.zeros(3, dtype=torch.long),
            update_mask=torch.tensor([1.0, 0.0]),
            token_tags=torch.zeros(3, dtype=torch.long),
            prompt_embeds=torch.zeros(1, 2),
            img_pos_info=torch.tensor([0, 1]),
            audio_pos_info=torch.tensor([2]),
            text_pos_info=torch.tensor([], dtype=torch.long),
            img_pos_for_infer_output_info=torch.tensor([0, 1]),
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, 3]), "max_seqlen_q": 3},
            refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, 3]), "max_seqlen_q": 3},
            rope_table=torch.zeros(3, 2),
            update_audio_mask=torch.tensor([0.0]),
        )
        namespace = dict(
            torch=torch,
            math=math,
            Any=Any,
            _FORWARD_SUPPORTED_KWARGS=set(inputs),
            _required_kwarg=lambda data, key: data[key],
            MINIMAX_H3_ADALN_MODALITY_NUM=3,
        )
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        model = SimpleNamespace(
            vdn_config=None,
            _pos_ids=lambda value, name: value,
            _psp_field=lambda value, name, key: value[key],
            _psp_optional=lambda value, key, default: value.get(key, default),
            _rope_local_span=lambda length: (0, length),
            _validate_prepared_rope_table=lambda *args, **kwargs: None,
            _embed=lambda **kwargs: (torch.ones(3, 4), torch.zeros(1, 2)),
            sp_prepare=lambda *args: args,
            sp_gather=lambda hidden: hidden,
            final_layer=lambda hidden, **kwargs: (hidden[:, :2], hidden[:, 2:]),
        )
        hidden, arguments, context = namespace["prepare_transformer_forward"](model, **inputs)
        original = weakref.ref(hidden)
        completed = hidden + 5
        del hidden, arguments, inputs
        self.assertIsNone(original())
        video, audio = namespace["finish_transformer_forward"](model, completed, **context)
        torch.testing.assert_close(video, torch.tensor([[6.0, 6.0], [0.0, 0.0]]))
        torch.testing.assert_close(audio, torch.zeros(1, 2))
