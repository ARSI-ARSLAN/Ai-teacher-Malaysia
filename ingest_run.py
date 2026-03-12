"""
ingest_run.py - One-shot CLI script to run the full PDF ingestion pipeline
Usage:
    python ingest_run.py              # Full ingestion → uploads to Zilliz
    python ingest_run.py --dry-run    # Test only, no upload
    python ingest_run.py --pdf-dir "D:/another/folder"
"""

import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="KSSM Math PDF Ingestion Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process PDFs but do NOT upload to Zilliz Cloud",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directory containing KSSM PDF files (default: current dir)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  AI Teacher Malaysia - PDF Ingestion Pipeline")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE: PDFs will be processed but NOT uploaded.")

    try:
        from backend.config import config
        from backend.ingest import run_ingestion

        # Validate config (skip Zilliz check for dry run)
        if not args.dry_run:
            config.validate()
            from backend.vector_store import init_collection
            logger.info("✅ Configuration validated.")

        # Run ingestion
        collection = None
        if not args.dry_run:
            collection = init_collection()

        stats = run_ingestion(
            pdf_dir=args.pdf_dir or config.PDF_DIR,
            collection=collection,
            dry_run=args.dry_run,
        )

        logger.info("\n" + "=" * 60)
        logger.info("  ✅ INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Files processed : {len(stats['files_processed'])}")
        for f in stats["files_processed"]:
            logger.info(f"    ✓ {f}")
        logger.info(f"  Total chunks    : {stats['total_chunks']:,}")
        logger.info(f"  Mode            : {'DRY RUN' if args.dry_run else '✅ UPLOADED'}")
        logger.info("=" * 60)

        if not args.dry_run:
            logger.info("\n🎉 Ready! Start the server with: python start.py")

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("   Make sure PDF files are in the project directory.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.error("   Please copy .env.example to .env and fill in credentials.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
