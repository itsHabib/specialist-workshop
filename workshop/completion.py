"""Model transport decoding, separate from task schema and semantic grading."""
import re

VERSION='empty-think-v1'


def decode(raw):
    # This pinned Qwen tokenizer adds this empty assistant channel during SFT.
    # Strip only the empty channel at the beginning, never prose/reasoning/JSON errors.
    match=re.match(r'\A<think>\s*</think>\s*(?=\{)',raw)
    if match:return dict(content=raw[match.end():],transport=VERSION,empty_channel_removed=True)
    return dict(content=raw,transport=VERSION,empty_channel_removed=False)
