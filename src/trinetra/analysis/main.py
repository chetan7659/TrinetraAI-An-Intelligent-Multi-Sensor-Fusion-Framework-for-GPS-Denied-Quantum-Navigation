"""Entry point for Exploratory Data Analysis (EDA) on the RoNIN dataset.

Usage:
    python -m trinetra.analysis.main
"""

from __future__ import annotations

import logging
from pathlib import Path

from trinetra.adapters.datasets.ronin import (
    RoninAdapter,
    RoninCanonicalMapper,
    RoninHDF5Reader,
    RoninMetadataLoader,
)
from trinetra.analysis.report_export import export_reports
from trinetra.analysis.summary import generate_statistics
from trinetra.application.dataset.recording_iterator import RecordingIterator
from trinetra.application.dataset.split_loader import SplitLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the EDA pipeline for the RoNIN dataset."""
    logger.info("Initializing RoNIN dataset services...")

    # 1. Initialize Infrastructure / Adapter Layer
    try:
        adapter = RoninAdapter()
    except Exception as e:
        logger.error(f"Failed to initialize RoninAdapter: {e}")
        return

    # 2. Initialize Application Layer (Recording Orchestrator)
    recording_iterator = RecordingIterator(
        metadata_loader=RoninMetadataLoader(),
        hdf5_reader=RoninHDF5Reader(),
        canonical_mapper=RoninCanonicalMapper(),
    )

    # 3. Initialize Application Layer (Split Orchestrator)
    split_loader = SplitLoader(adapter, recording_iterator)

    # 4. Generate Statistics (Analysis Layer)
    # Dynamically discover the dataset splits using the adapter
    splits_to_analyze = adapter.list_splits()

    if not splits_to_analyze:
        logger.warning("No dataset splits found. Aborting analysis.")
        return

    logger.info(f"Starting single-pass streaming analysis on splits: {splits_to_analyze}")
    analysis_result = generate_statistics(split_loader, splits_to_analyze)

    # 5. Export Reports
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = project_root / "reports" / "eda" / "ronin" / "statistics"

    logger.info(f"Exporting analysis reports to {output_dir}")
    export_reports(analysis_result, output_dir)

    # 6. Generate Visualizations & Quality Assessment (Analysis Layer)
    figures_dir = project_root / "reports" / "eda" / "ronin" / "figures"
    quality_dir = project_root / "reports" / "eda" / "ronin" / "quality"

    from trinetra.analysis.plot_utils import save_figure
    from trinetra.analysis.quality_assessment import assess_recording_quality
    from trinetra.analysis.quality_summary import aggregate_quality_results, export_quality_reports
    from trinetra.analysis.visualization import generate_recording_plots, plot_dataset_histograms

    logger.info("Generating dataset histograms...")
    histograms = plot_dataset_histograms(analysis_result)
    for fig, name in zip(
        histograms,
        ["recording_duration_histogram.png", "sampling_frequency_histogram.png"],
        strict=False,
    ):
        save_figure(fig, figures_dir / name)

    logger.info(
        "Starting recording-level quality assessment, geometry validation, and time-series plotting..."
    )

    quality_results = []
    validation_results = []

    for split in splits_to_analyze:
        try:
            recordings = list(split_loader._adapter.list_recordings(split))
            if not recordings:
                continue

            for idx, rec in enumerate(recordings):
                rec_id = getattr(rec, "id", "unknown")

                # Assess quality for every recording
                stream = split_loader._recording_iterator.iter_recording(rec)
                q_res = assess_recording_quality(rec_id, stream)
                quality_results.append(q_res)

                # Validate geometry for every recording
                from trinetra.analysis.validation_summary import (
                    aggregate_validation_results,
                    export_validation_reports,
                    validate_recording_geometry,
                )

                stream_for_val = split_loader._recording_iterator.iter_recording(rec)
                v_res = validate_recording_geometry(rec_id, stream_for_val)
                validation_results.append(v_res)

                # Plotting time-series only for the first recording in each split as a representative
                if idx == 0:
                    logger.info(
                        f"Plotting time-series for representative recording {rec_id} from split {split}..."
                    )

                    # We need to recreate the iterator since the quality assessment consumed it
                    stream_for_plot = split_loader._recording_iterator.iter_recording(rec)
                    plots = generate_recording_plots(stream_for_plot)

                    rec_figures_dir = figures_dir / rec_id
                    for filename, fig in plots.items():
                        save_figure(fig, rec_figures_dir / filename)

        except Exception as e:
            logger.error(f"Failed to process recordings for split {split}: {e}")

    logger.info("Aggregating quality reports...")
    quality_dfs = aggregate_quality_results(quality_results)
    export_quality_reports(quality_dfs, quality_dir)

    logger.info("Aggregating coordinate and orientation validation reports...")
    validation_dir = project_root / "reports" / "eda" / "ronin" / "coordinate_validation"
    validation_dfs = aggregate_validation_results(validation_results)
    export_validation_reports(validation_dfs, validation_dir)

    logger.info("EDA pipeline completed successfully.")


if __name__ == "__main__":
    main()
