import argparse
import asyncio
import os
import sys

# This is a bit of a hack to allow this script to be run from the command line
# and still have access to the rest of the application's modules.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from app.utils.cache import CacheManager
from app.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


async def main():
    """
    Main function to parse arguments and flush the cache.
    """
    parser = argparse.ArgumentParser(
        description="Flush Redis cache keys matching a pattern."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Pattern to flush (e.g., 'dynamic_q:*'). Wildcards supported.",
    )
    args = parser.parse_args()

    logger.info(f"Attempting to flush cache with pattern: {args.pattern}")

    try:
        deleted_count = await CacheManager.flush_by_pattern(args.pattern)
        logger.info(
            f"Successfully flushed {deleted_count} keys matching "
            f"pattern '{args.pattern}'."
        )
    except Exception as e:
        logger.error(f"An error occurred while flushing the cache: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # To run this script:
    # python -m app.tools.flush_cache --pattern "dynamic_q:*"
    asyncio.run(main())
