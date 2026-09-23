"""Source-level regression test for the MUSA ViT backend timing fix."""

from pathlib import Path


PLATFORM = Path(__file__).parents[1] / "vllm_musa" / "platform.py"


def test_vit_selector_registers_musa_backends_before_candidate_walk():
    source = PLATFORM.read_text(encoding="utf-8")
    start = source.index("    def get_vit_attn_backend(")
    end = source.index("    @classmethod", start + 10)
    body = source[start:end]
    assert body.index("register_attention_backends()") < body.index(
        "if backend is not None"
    )
    assert body.index("register_attention_backends()") < body.index(
        "for vit_attn_backend in cls.get_supported_vit_attn_backends()"
    )
