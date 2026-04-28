"""
Training scheduler for the Self-Improving RAG system.

Uses APScheduler to trigger the CrossEncoder fine-tuning process
periodically (e.g., nightly).
"""

import logging
import asyncio
from typing import Optional, Tuple
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from self_improving_rag.training.trainer import train_reranker

logger = logging.getLogger(__name__)

# Module-level scheduler instance
_scheduler = AsyncIOScheduler()


async def scheduled_train_job() -> None:
    """
    Wrapper for the trainer to be called by the scheduler.
    """
    logger.info("Triggering scheduled fine-tuning job...")
    try:
        checkpoint = await train_reranker()
        if checkpoint:
            logger.info(f"Scheduled training successful: {checkpoint}")
        else:
            logger.info("Scheduled training skipped (not enough data).")
    except Exception as exc:
        logger.error(f"Scheduled training job failed: {exc}")


def start_scheduler(hour: int = 2, minute: int = 0, interval_minutes: Optional[int] = None) -> None:
    """
    Start the background scheduler for training.

    Args:
        hour: Hour of the day to run (if using Cron).
        minute: Minute of the hour (if using Cron).
        interval_minutes: If provided, runs every X minutes instead of nightly.
    """
    if _scheduler.running:
        logger.warning("Scheduler is already running.")
        return

    if interval_minutes:
        from apscheduler.triggers.interval import IntervalTrigger
        _scheduler.add_job(
            scheduled_train_job,
            IntervalTrigger(minutes=interval_minutes),
            id="interval_training",
            replace_existing=True
        )
        logger.info(f"Training scheduler started (runs every {interval_minutes} minutes).")
    else:
        _scheduler.add_job(
            scheduled_train_job,
            CronTrigger(hour=hour, minute=minute),
            id="nightly_training",
            replace_existing=True
        )
        logger.info(f"Nightly training scheduler started (runs at {hour:02d}:{minute:02d}).")

    _scheduler.start()


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    if _scheduler.running:
        _scheduler.shutdown()
        logger.info("Training scheduler stopped.")
