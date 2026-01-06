# scripts/run_data_pipeline.py
import logging
from pathlib import Path

from app.data_acquisition import acquire_and_prepare_data
from app.indexer import build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Default locations come from app.config via acquire_and_prepare_data() and build_index()
    # If you want to override sources/txt/pdf path, pass them to acquire_and_prepare_data(...)
    try:
        logger.info("Starting data acquisition (scrape -> convert)...")
        pdf_path = acquire_and_prepare_data(overwrite=False)  # set overwrite=True to force re-acquire & re-convert
        logger.info("Data acquisition complete. PDF created at: %s", pdf_path)

        logger.info("Starting index build from PDF...")
        # build_index uses defaults from app.config when no args provided
        vs = build_index()
        logger.info("Index build complete. Vectorstore created.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise

if __name__ == "__main__":
    main()