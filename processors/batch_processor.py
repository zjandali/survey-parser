"""
Batch processor for handling multiple PDF files.
"""

import os
import sys
import time
import uuid
import asyncio
import signal
import logging
import json
import glob
from typing import List, Dict, Any, Tuple, Set, Optional

from config.settings import (
    OUTPUT_DIR,
    TOGETHER_CONCURRENCY_LIMIT,
    MISTRAL_CONCURRENCY_LIMIT,
    OPENROUTER_CONCURRENCY_LIMIT,
    DEEPSEEK_CONCURRENCY_LIMIT,
    CLAUDE_CONCURRENCY_LIMIT,
    GEMINI_CONCURRENCY_LIMIT
)
from core.file_operations import create_directory_async
from core.resource_manager import check_resources, print_summary
from processors.single_processor import SingleFileProcessor

# Use existing logger instead of creating a duplicate
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processor for handling multiple PDF files in batch"""
    
    def __init__(self, llm_provider: str = "mistral"):
        """
        Initialize the batch processor
        
        Args:
            llm_provider: LLM provider to use ("mistral", "together", "openrouter", "deepseek", "claude", or "gemini")
        """
        self.single_processor = SingleFileProcessor(llm_provider=llm_provider)
        self.running_tasks: Set[str] = set()
        self.canceled = False
        self.start_time = None
        self.llm_provider = llm_provider
        
        # Set default concurrency based on provider
        if llm_provider == "together":
            self.default_concurrency = TOGETHER_CONCURRENCY_LIMIT
        elif llm_provider == "openrouter":
            self.default_concurrency = OPENROUTER_CONCURRENCY_LIMIT
        elif llm_provider == "deepseek":
            self.default_concurrency = DEEPSEEK_CONCURRENCY_LIMIT
        elif llm_provider == "claude":
            self.default_concurrency = CLAUDE_CONCURRENCY_LIMIT
        elif llm_provider == "gemini":
            self.default_concurrency = GEMINI_CONCURRENCY_LIMIT
        else:  # mistral or fallback
            self.default_concurrency = MISTRAL_CONCURRENCY_LIMIT
        
        logger.info(f"Initialized batch processor with LLM provider: {llm_provider}")
    
    def handle_signals(self, signum, frame):
        """
        Handle interrupt signals gracefully
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if not self.canceled:
            logger.warning("Received interrupt signal. Canceling operations (this may take a moment)...")
            self.canceled = True
        else:
            logger.warning("Second interrupt received. Exiting immediately.")
            sys.exit(1)
    
    async def process_file_with_resource_management(
        self, 
        pdf_path: str, 
        output_dir: str,
        timeout: int, 
        max_ram_percent: float
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Process a single PDF file with resource management
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results
            timeout: Timeout in seconds for OCR processing
            max_ram_percent: Maximum RAM usage percent threshold
            
        Returns:
            Tuple of (file_path, success, error_message)
        """
        if self.canceled:
            return pdf_path, False, "Canceled by user"
        
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Create temp directory for processing
        temp_dir = os.path.join(output_dir, "temp")
        await create_directory_async(temp_dir)
        
        # Create ocr and prompts directories
        ocr_dir = os.path.join(output_dir, "ocr")
        prompts_dir = os.path.join(output_dir, "prompts")
        await create_directory_async(ocr_dir)
        await create_directory_async(prompts_dir)
        
        try:
            # Simply use the base filename for output
            mutation_file = f"{file_base}.json"
            mutation_path = os.path.join(output_dir, mutation_file)
            
            logger.info(f"Starting processing of {file_name}")
            
            # Create a unique temporary directory for intermediate files
            unique_id = str(uuid.uuid4())[:8]
            file_temp_dir = os.path.join(temp_dir, f"{file_base}_{unique_id}")
            await create_directory_async(file_temp_dir)
            
            # Process the PDF asynchronously
            _, _, structured_data, processed_pages, provider = await self.single_processor.process_pdf(
                pdf_path,
                output_dir=output_dir,  # Use main output directory
                # We still need to generate these files for process_pdf_async, but we won't keep them
                output_file="temp_structured_results.json",  
                mutation_file="temp_mutation.json",
                wait_timeout=timeout
            )
            
            # Read the mutation JSON data from the main output directory
            temp_mutation_path = os.path.join(output_dir, "temp_mutation.json")
            
            # If the file exists, read it and save to the final destination
            if os.path.exists(temp_mutation_path):
                with open(temp_mutation_path, 'r') as src_file:
                    mutation_data = json.load(src_file)
                    
                # Write the mutation data directly to the output directory
                with open(mutation_path, 'w') as dest_file:
                    json.dump(mutation_data, dest_file, indent=2)
                    
                logger.info(f"Successfully processed {file_name} - Saved to {mutation_path}")
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(file_temp_dir, ignore_errors=True)
                
                return pdf_path, True, None
            else:
                error_message = f"Failed to create mutation file for {file_name}"
                logger.error(error_message)
                return pdf_path, False, error_message
        
        except Exception as e:
            error_message = f"Error processing {file_name}: {str(e)}"
            logger.error(error_message)
            return pdf_path, False, error_message
        
        finally:
            # Clean up memory and release resources
            if pdf_path in self.running_tasks:
                self.running_tasks.remove(pdf_path)
    
    async def batch_process_pdfs(
        self,
        input_files: List[str],
        output_dir: str = OUTPUT_DIR,
        concurrency: Optional[int] = None,
        timeout: int = 300,
        max_ram_percent: float = 80.0
    ) -> Dict[str, Any]:
        """
        Process multiple PDF files in parallel with resource management
        
        Args:
            input_files: List of PDF file paths to process
            output_dir: Directory to save results
            concurrency: Maximum number of files to process simultaneously
            timeout: Timeout in seconds for OCR processing
            max_ram_percent: Maximum RAM usage percent threshold
            
        Returns:
            Dictionary with processing results
        """
        # Use default concurrency if none specified
        if concurrency is None:
            concurrency = self.default_concurrency
        
        # Enforce maximum concurrency based on provider
        if self.llm_provider == "together" and concurrency > TOGETHER_CONCURRENCY_LIMIT:
            concurrency = TOGETHER_CONCURRENCY_LIMIT
        elif self.llm_provider == "openrouter" and concurrency > OPENROUTER_CONCURRENCY_LIMIT:
            concurrency = OPENROUTER_CONCURRENCY_LIMIT
        elif self.llm_provider == "deepseek" and concurrency > DEEPSEEK_CONCURRENCY_LIMIT:
            concurrency = DEEPSEEK_CONCURRENCY_LIMIT
        elif self.llm_provider == "claude" and concurrency > CLAUDE_CONCURRENCY_LIMIT:
            concurrency = CLAUDE_CONCURRENCY_LIMIT
        elif self.llm_provider == "gemini" and concurrency > GEMINI_CONCURRENCY_LIMIT:
            concurrency = GEMINI_CONCURRENCY_LIMIT
        
        # Create output directory
        await create_directory_async(output_dir)
        
        # Start resource monitoring in background
        monitor_task = asyncio.create_task(
            check_resources(
                running_tasks=self.running_tasks,
                canceled=self.canceled,
                interval=5.0, 
                max_ram_percent=max_ram_percent
            )
        )
        
        self.start_time = time.time()
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(file_path):
            """Process a file while respecting the concurrency limit"""
            async with semaphore:
                self.running_tasks.add(file_path)
                result = await self.process_file_with_resource_management(
                    file_path, output_dir, timeout, max_ram_percent
                )
                return result
        
        # Create tasks for all files
        tasks = []
        for file_path in input_files:
            if self.canceled:
                break
            
            task = asyncio.create_task(process_with_semaphore(file_path))
            tasks.append(task)
        
        # Process results as they complete
        for future in asyncio.as_completed(tasks):
            if self.canceled:
                break
                
            try:
                file_path, success, error = await future
                if success:
                    results["success"].append(file_path)
                else:
                    results["failed"].append((file_path, error))
            except Exception as e:
                logger.error(f"Error in task: {str(e)}")
        
        # Cancel resource monitor
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        results["elapsed_time"] = elapsed_time
        results["success_count"] = len(results["success"])
        results["failed_count"] = len(results["failed"])
        results["skipped_count"] = len(results["skipped"])
        results["total_count"] = results["success_count"] + results["failed_count"] + results["skipped_count"]
        results["output_dir"] = output_dir
        
        return results
    
    async def process_directory(
        self,
        input_dir: str,
        concurrency: Optional[int] = None,
        timeout: int = 300,
        pattern: str = "*.pdf",
        max_ram_percent: float = 80.0
    ) -> Dict[str, Any]:
        """
        Process all PDF files in a directory
        
        Args:
            input_dir: Directory containing PDF files
            concurrency: Maximum number of files to process simultaneously
            timeout: Timeout in seconds for OCR processing
            pattern: File pattern to match
            max_ram_percent: Maximum RAM usage percent threshold
            
        Returns:
            Dictionary with processing results
        """
        # Use default concurrency if none specified
        if concurrency is None:
            concurrency = self.default_concurrency
            
        # Enforce maximum concurrency based on provider - simplified to remove redundant logging
        if self.llm_provider == "together" and concurrency > TOGETHER_CONCURRENCY_LIMIT:
            concurrency = TOGETHER_CONCURRENCY_LIMIT
        elif self.llm_provider == "openrouter" and concurrency > OPENROUTER_CONCURRENCY_LIMIT:
            concurrency = OPENROUTER_CONCURRENCY_LIMIT
        elif self.llm_provider == "deepseek" and concurrency > DEEPSEEK_CONCURRENCY_LIMIT:
            concurrency = DEEPSEEK_CONCURRENCY_LIMIT
        elif self.llm_provider == "claude" and concurrency > CLAUDE_CONCURRENCY_LIMIT:
            concurrency = CLAUDE_CONCURRENCY_LIMIT
        elif self.llm_provider == "gemini" and concurrency > GEMINI_CONCURRENCY_LIMIT:
            concurrency = GEMINI_CONCURRENCY_LIMIT
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_signals)
        signal.signal(signal.SIGTERM, self.handle_signals)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            logger.error(f"Input directory '{input_dir}' does not exist.")
            return {"error": f"Input directory '{input_dir}' does not exist."}
            
        # Get list of PDF files to process
        if ',' in pattern:
            # Handle comma-separated patterns
            patterns = pattern.split(',')
            input_files = []
            for p in patterns:
                p = p.strip()
                pattern_path = os.path.join(input_dir, p)
                input_files.extend(glob.glob(pattern_path))
        else:
            # Single pattern
            pattern_path = os.path.join(input_dir, pattern)
            input_files = glob.glob(pattern_path)
        
        # Sort and remove duplicates
        input_files = sorted(set(input_files))
        
        if not input_files:
            logger.error(f"No files matching '{pattern}' found in '{input_dir}'.")
            return {"error": f"No files matching '{pattern}' found in '{input_dir}'."}
            
        logger.info(f"Found {len(input_files)} files to process")
        logger.info(f"Concurrency limit: {concurrency}")
        logger.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        
        # Process the files in batch
        results = await self.batch_process_pdfs(
            input_files,
            OUTPUT_DIR,
            concurrency=concurrency,
            timeout=timeout,
            max_ram_percent=max_ram_percent
        )
        
        # Print summary
        print_summary(results)
        
        # Save results summary
        summary_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
        with open(summary_path, 'w') as f:
            # Convert paths to strings and filter out non-serializable data
            serializable_results = {
                "success": [str(p) for p in results["success"]],
                "failed": [(str(p), e) for p, e in results["failed"]],
                "skipped": [str(p) for p in results.get("skipped", [])],
                "elapsed_time": results["elapsed_time"],
                "success_count": results["success_count"],
                "failed_count": results["failed_count"],
                "skipped_count": results["skipped_count"],
                "total_count": results["total_count"]
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        return results 