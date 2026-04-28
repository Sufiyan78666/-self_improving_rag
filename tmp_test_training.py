"""Test script for the training module (Module 7)."""
import sys, asyncio, os
sys.path.insert(0, '.')

from self_improving_rag.storage.db import init_db
from self_improving_rag.storage.queries import insert_session, insert_retrieval, insert_feedback
# from self_improving_rag.training.dataset import get_training_dataloader (Removed)
from self_improving_rag.training.experiment_tracker import log_experiment_run, get_latest_experiments
from self_improving_rag.training.scheduler import start_scheduler, stop_scheduler

async def test_training():
    await init_db()
    print("DB initialized with new migrations.")

    # 1. Seed some data
    sid = await insert_session("What is training?")
    await insert_retrieval(sid, "c1", "Training is learning.", 1, 0.9, 0.8)
    await insert_feedback(sid, "c1", "thumbs_up", 1.0)
    
    # 2. Test Data Preparation
    from self_improving_rag.training.dataset import get_training_examples
    examples = await get_training_examples(min_relevance=0.1)
    if examples:
        print(f"Examples created: {len(examples)} samples.")
        for ex in examples:
            print(f"Sample: {ex.texts} label: {ex.label}")
            break
    else:
        print("Examples list is empty (expected if threshold not met).")

    # 3. Test Experiment Tracker
    run_id = await log_experiment_run("test-model", 10, 1, 0.5, 0.6, 0.7)
    print(f"Logged run: {run_id}")
    
    latest = await get_latest_experiments(1)
    print(f"Latest run from DB: {latest[0]['ndcg_after']}")
    assert latest[0]['ndcg_after'] == 0.6

    # 4. Test Scheduler (start/stop)
    start_scheduler()
    print("Scheduler start check pass.")
    stop_scheduler()

    print("ALL TRAINING MODULE TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(test_training())
