from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    csv_path: str = "df_fixed_with_cropped_roi.csv"
    id_col: str = "id"
    fold_col: str = "fold"
    target_cols: list[str] = field(default_factory=lambda: [
        "post_PTA_0.5k_A", "post_PTA_1k_A", "post_PTA_2k_A", "post_PTA_3k_A",
        "post_PTA_0.5k_B", "post_PTA_1k_B", "post_PTA_2k_B", "post_PTA_3k_B",
    ])
    roi_cols: list[str] = field(default_factory=lambda: ["roi_path_25_sphere", "roi_path_40_sphere", "roi_path_60_sphere"])
    out_dir: str = "results/main_experiment"
    seed: int = 42
    device: str | None = None
    use_amp: bool = True
