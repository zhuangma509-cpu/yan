"""Quick test for SCUT split parsing (no dataset download required)."""

from __future__ import annotations

from data.scut_fbp5500 import map_scut_score_to_training_scale, parse_split_file_line


def main() -> None:
    assert parse_split_file_line("mty152.jpg 4.316667") == ("mty152.jpg", 4.316667)
    assert parse_split_file_line("  fty276.jpg  2.133333  ") == ("fty276.jpg", 2.133333)
    assert parse_split_file_line("# comment") is None
    assert map_scut_score_to_training_scale(1.0, "linear_1_to_100") == 1.0
    assert map_scut_score_to_training_scale(5.0, "linear_1_to_100") == 100.0
    assert map_scut_score_to_training_scale(3.0, "round_1_to_5") == 3.0
    print("test_scut_fbp5500_parse OK")


if __name__ == "__main__":
    main()
