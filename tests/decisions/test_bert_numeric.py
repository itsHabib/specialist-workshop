"""Exercise the numerical backend with a generated encoder and a local WordPiece vocabulary."""

from importlib.util import find_spec
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


HAS_BERT_STACK = all(find_spec(package) is not None for package in
                     ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_BERT_STACK, "optional BERT training dependencies are not installed")
class BertNumericalTest(unittest.TestCase):
    def test_real_gradients_local_reload_and_input_boundaries(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        from decision_workshop.bert import fit_bert, load_scorer

        task = {
            "name": "route", "features": {"note": "text"}, "labels": ["close", "retry"],
            "question": "which action", "descriptions": {"close": "done", "retry": "again"},
        }
        rows = [
            {"id": str(i), "split": "train", "state": {"note": note}, "target": target}
            for i, (note, target) in enumerate([
                ("done", "close"), ("again", "retry"), ("done done", "close"),
            ])
        ]
        recipe = dict(epochs=2, batch_size=2, learning_rate=0.001, seed=91)
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            out = Path(tmp) / "checkpoint"
            base.mkdir()
            out.mkdir()
            vocab = [
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "note", ":", "which",
                "action", "option", "close", "retry", ".", "done", "again",
            ]
            vocab_path = base / "vocab.txt"
            vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
            BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(base)
            config = BertConfig(
                vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            )
            torch.manual_seed(10)
            initial = BertModel(config, add_pooling_layer=False)
            initial.save_pretrained(base, safe_serialization=True)

            with self.assertRaisesRegex(ValueError, "train-only"):
                fit_bert(task, rows + [dict(rows[0], split="test")], base, out, **recipe)
            self.assertEqual(list(out.iterdir()), [])
            stats = fit_bert(task, rows, base, out, **recipe)
            self.assertEqual(stats["optimizer_steps"], 4)
            self.assertNotEqual(stats["backbone_sha256_before"], stats["backbone_sha256_after"])
            self.assertTrue(math.isfinite(stats["initial_loss"]) and math.isfinite(stats["final_loss"]))

            # Weight decay alone cannot move a zero bias away from zero. Read
            # the encoder directly from disk to verify it received a gradient.
            trained = BertModel.from_pretrained(out, add_pooling_layer=False, local_files_only=True,
                                                use_safetensors=True)
            self.assertEqual(torch.count_nonzero(initial.embeddings.LayerNorm.bias).item(), 0)
            self.assertGreater(torch.count_nonzero(trained.embeddings.LayerNorm.bias).item(), 0)
            scorer = load_scorer(out, task)
            state = {"note": "again again"}
            scores = scorer(state)
            self.assertEqual(set(scores), set(task["labels"]))
            self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)
            self.assertTrue(all(0.0 < p < 1.0 for p in scores.values()))
            self.assertEqual(scores, load_scorer(out, task)(state))
            with self.assertRaisesRegex(ValueError, "32-token limit"):
                scorer({"note": "again " * 100})
