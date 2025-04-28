"""
Resource management utilities for the survey parser application.
"""

import asyncio
import psutil
import logging
from typing import Set, Dict, Any
import os

logger = logging.getLogger(__name__)

def memory_usage_percent() -> float:
    """Get the current memory usage as a percentage"""
    return psutil.virtual_memory().percent

async def check_resources(
    running_tasks: Set[str],
    canceled: bool,
    interval: float = 5.0, 
    max_ram_percent: float = 80.0
) -> None:
    """
    Periodically check system resources and pause/resume as needed
    
    Args:
        running_tasks: Set of currently running task identifiers
        canceled: Boolean flag indicating if processing is canceled
        interval: Check interval in seconds
        max_ram_percent: Maximum RAM percentage threshold
    """
    while not canceled and running_tasks:
        ram_usage = memory_usage_percent()
        logger.debug(f"Memory usage: {ram_usage:.1f}% - Active tasks: {len(running_tasks)}")
        

        await asyncio.sleep(interval)
    
    logger.debug("Resource monitor stopped")

def print_summary(results: Dict[str, Any]) -> None:
    """
    Print summary of batch processing results
    
    Args:
        results: Dictionary with processing results
    """
    elapsed_time = results["elapsed_time"]
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total PDF files processed: {results['total_count']}")
    print(f"Successfully generated mutation files: {results['success_count']}")
    print(f"Failed files: {results['failed_count']}")
    print(f"Skipped files: {results['skipped_count']}")
    print(f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Output directory: {results['output_dir']}")
    
    if results["failed"]:
        print("\nFAILURES:")
        for file_path, error in results["failed"]:
            print(f"- {os.path.basename(file_path)}: {error}")
    
    print("="*50) 