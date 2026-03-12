"""
start.py - Start the AI Teacher Malaysia FastAPI server
Usage: python start.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting AI Teacher Malaysia server...")

    try:
        from backend.config import config
        import uvicorn

        print(
            f"""
╔══════════════════════════════════════════════════════╗
║        AI Teacher Malaysia - RAG System              ║
║        AI Guru Matematik KSSM Tingkatan 1-5          ║
╠══════════════════════════════════════════════════════╣
║  Server  : http://localhost:{config.PORT}                  ║
║  API     : http://localhost:{config.PORT}/api/health       ║
║  Model   : {config.GROQ_MODEL:<40}  ║
╚══════════════════════════════════════════════════════╝
"""
        )

        uvicorn.run(
            "backend.api:app",
            host=config.HOST,
            port=config.PORT,
            reload=False,
            log_level="info",
        )

    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("   Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
