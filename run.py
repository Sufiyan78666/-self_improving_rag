"""
Main entry point for the Self-Improving RAG system.

Supports running the Streamlit UI or the Nightly Trainer.
Usage:
    python run.py ui      # Start the Streamlit app
    python run.py train   # Run a manual training cycle
"""

import sys
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)

logger = logging.getLogger("rag_run")


def start_ui():
    """Start the Streamlit UI."""
    import os
    logger.info("Starting Streamlit UI...")
    # Use sys.executable to ensure we use the same python environment
    cmd = f'"{sys.executable}" -m streamlit run self_improving_rag/ui/app.py'
    os.system(cmd)


async def run_train():
    """Run a manual training cycle."""
    from self_improving_rag.training.trainer import train_reranker
    logger.info("Starting manual training cycle...")
    try:
        checkpoint = await train_reranker()
        if checkpoint:
            print(f"✅ Training successful. New model: {checkpoint}")
        else:
            print("ℹ️ Training skipped: insufficient feedback.")
    except Exception as e:
        print(f"FAILED Training failed: {e}")
        print(f"❌ Training failed: {e}")


async def run_scheduler():
    """Start the background scheduler and keep the process alive."""
    from self_improving_rag.training.scheduler import start_scheduler
    # Check every 60 minutes for the 50-feedback threshold
    start_scheduler(interval_minutes=60)
    print("🚀 Training scheduler is running in the background...")
    print("It will check every 60 minutes if the 50-feedback threshold is met.")
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        from self_improving_rag.training.scheduler import stop_scheduler
        stop_scheduler()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [ui|train|schedule]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "ui":
        start_ui()
    elif cmd == "train":
        asyncio.run(run_train())
    elif cmd == "schedule":
        asyncio.run(run_scheduler())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
