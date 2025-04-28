"""
Processor for handling a single PDF file.
"""

import os
import json
import uuid
import asyncio
import logging
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from config.settings import (
    OUTPUT_DIR,
    TOGETHER_REQUEST_DELAY

)
from models.enums import FeasibilityProvider
from core.file_operations import (
    create_directory_async, 
    write_file_async, 
    rmtree_async
)
from core.pdf_utils import get_pdf_page_count_async
from services.ocr_service import OCRService
from services.provider_service import ProviderService
from services.extraction_service import ExtractionService

class SingleFileProcessor:
    """Processor for handling a single PDF file"""
    
    def __init__(self, llm_provider: str = "mistral"):
        """
        Initialize the single file processor
        
        Args:
            llm_provider: LLM provider to use ("mistral", "together", "openrouter", "deepseek", "claude", or "gemini")
        """
        self.ocr_service = OCRService()
        self.provider_service = ProviderService(llm_provider=llm_provider)
        self.extraction_service = ExtractionService(llm_provider=llm_provider)
        self.llm_provider = llm_provider
        
        # Rate limiting parameters
        if llm_provider == "together":
            self.delay_between_requests = TOGETHER_REQUEST_DELAY
            self.concurrency_semaphore_limit = 3  # Allow 3 concurrent API calls for safety
        elif llm_provider == "openrouter":
            self.delay_between_requests = 1.0  # More conservative delay for OpenRouter
            self.concurrency_semaphore_limit = 2  # More conservative limit for OpenRouter
        elif llm_provider == "deepseek":
            self.delay_between_requests = 1.0  # Same conservative delay for DeepSeek
            self.concurrency_semaphore_limit = 2  # Same conservative limit for DeepSeek
        elif llm_provider == "claude":
            self.delay_between_requests = 1.0  # Same conservative delay for Claude
            self.concurrency_semaphore_limit = 2  # Same conservative limit for Claude
        elif llm_provider == "gemini":
            self.delay_between_requests = 1.0  # Same conservative delay for Gemini
            self.concurrency_semaphore_limit = 2  # Same conservative limit for Gemini
        else:  # mistral
            self.delay_between_requests = 0
            self.concurrency_semaphore_limit = 5
        
        logging.info(f"Initialized single file processor with LLM provider: {llm_provider}")
    
    async def process_single_page(
        self,
        pdf_path: str, 
        page_num: int, 
        temp_dir: str, 
        session: aiohttp.ClientSession, 
        wait_timeout: int = 300
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
        """
        Process a single page of the PDF
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process
            temp_dir: Directory for temporary files
            session: aiohttp session
            wait_timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (page_number, ocr_text, extracted_json_data)
        """
        # Extract text
        ocr_text = await self.ocr_service.extract_text_from_page(
            pdf_path, page_num, temp_dir, session, wait_timeout
        )
        
        if not ocr_text or len(ocr_text.strip()) < 100:  # Arbitrary threshold for meaningful content
            print(f"Page {page_num} has insufficient text content")
            return page_num, ocr_text, None
        
        # Process text with Mistral
        _, json_data = await self.extraction_service.process_text_with_mistral(
            ocr_text, f"page_{page_num}"
        )
        
        # Add page number to each question for ordering purposes
        if json_data and "sections" in json_data:
            for section in json_data["sections"]:
                for question in section.get("questions", []):
                    # Add page number metadata to preserve ordering
                    question["_page_num"] = page_num
        
        return page_num, ocr_text, json_data
    
    async def process_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        output_file: str = "structured_survey_results.json",
        mutation_file: str = "mutation_format.json",
        wait_timeout: int = 300
    ) -> Tuple[str, str, str, List[int], FeasibilityProvider]:
        """
        Process a PDF file completely
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results (default: OUTPUT_DIR)
            start_page: First page to process (1-based index)
            end_page: Last page to process (1-based index, None for all pages)
            output_file: Filename for structured data
            mutation_file: Filename for mutation format data
            wait_timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (structured_path, mutation_path, structured_json, processed_pages, provider)
        """
        # Use the provided output_dir for all operations
        if not output_dir:
            output_dir = OUTPUT_DIR
        results_dir = Path(output_dir)
        await create_directory_async(results_dir)
        
        # Subdirs relative to the main output dir for this specific PDF run
        base_name = Path(pdf_path).stem
        run_temp_dir = results_dir / "temp" / f"{base_name}_{str(uuid.uuid4())[:8]}"
        intermediate_dir = run_temp_dir / "ocr_text"  # Store OCR within run's temp
        await create_directory_async(run_temp_dir)
        await create_directory_async(intermediate_dir)
        
        # Create ocr and prompts directories directly in the output directory
        ocr_dir = results_dir / "ocr"
        prompts_dir = results_dir / "prompts"
        await create_directory_async(ocr_dir)
        await create_directory_async(prompts_dir)
        
        pdf_path_obj = Path(pdf_path)  # Use Path object
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"File '{pdf_path}' does not exist.")
        
        # Determine if this is a PDF file or another supported format
        file_extension = pdf_path_obj.suffix.lower()
        if file_extension == '.pdf':
            total_pages = await get_pdf_page_count_async(pdf_path_obj)
            print(f"Document '{pdf_path_obj.name}' has {total_pages} pages")
            
            effective_end_page = min(end_page if end_page else total_pages, total_pages)
            pages_to_process = list(range(start_page, effective_end_page + 1))
        elif file_extension in ['.doc', '.docx']:
            # For Word documents, treat as a single page document
            print(f"Processing Word document '{pdf_path_obj.name}'")
            total_pages = 1
            effective_end_page = 1
            pages_to_process = [1]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .pdf, .doc, and .docx")
        
        if not pages_to_process:
            print(f"No pages to process for '{pdf_path_obj.name}' in range {start_page}-{effective_end_page}.")
            # Return empty/default structure to avoid downstream errors
            return (
                str(results_dir / output_file),  # Dummy paths
                str(results_dir / mutation_file), 
                json.dumps({"sections": []}), 
                [], 
                FeasibilityProvider.NULL
            )

        print(f"Processing pages {start_page} to {effective_end_page} ({len(pages_to_process)} pages total) for '{pdf_path_obj.name}'")
        
        combined_data = {"sections": []}
        processed_pages = []
        all_ocr_text = ""
        all_prompts = ""
        
        # Define output paths relative to the main output_dir passed to this function
        structured_output_path = results_dir / output_file
        mutation_output_path = results_dir / mutation_file
        ocr_output_path = ocr_dir / f"{base_name}.txt"
        prompts_output_path = prompts_dir / f"{base_name}.txt"

        # First, extract OCR text from all pages and store it
        page_texts = {}
        
        # Use a shared connector for efficiency within this process run
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Extract text from all pages first
            ocr_tasks = []
            for page_num in pages_to_process:
                ocr_task = asyncio.create_task(
                    self.ocr_service.extract_text_from_page(
                        pdf_path_obj, 
                        page_num, 
                        run_temp_dir,
                        session, 
                        wait_timeout
                    )
                )
                ocr_task.set_name(f"{base_name}_ocr_page_{page_num}")
                ocr_tasks.append((page_num, ocr_task))
            
            # Wait for all OCR tasks to complete
            for page_num, ocr_task in ocr_tasks:
                ocr_text = await ocr_task
                if ocr_text:
                    page_texts[page_num] = ocr_text
                    all_ocr_text += ocr_text + "\n\n"  # Collect all OCR text for provider identification
                    ocr_path = intermediate_dir / f"page_{page_num}_text.txt"
                    await write_file_async(ocr_path, ocr_text)
            
            # Save combined OCR text to output/ocr/pdfname.txt
            await write_file_async(ocr_output_path, all_ocr_text)
            
            # Now process pages using chunks that overlap across page boundaries
            # Each chunk will contain the current page plus context from adjacent pages
            
            # For Together API and OpenRouter, process pages with controlled parallelism to respect rate limits
            if self.llm_provider in ["together", "openrouter"]:
                # Create a semaphore to limit concurrent API calls to Together/OpenRouter
                api_semaphore = asyncio.Semaphore(self.concurrency_semaphore_limit)
                
                async def process_chunk_with_rate_limiting(page_num):
                    """Process a chunk with rate limiting for API calls"""
                    if page_num not in page_texts:
                        return page_num, None
                    
                    # Create a chunk containing content from adjacent pages
                    chunk_text = ""
                    
                    # Add last k lines of previous page if available (context before)
                    if page_num > start_page and (page_num - 1) in page_texts:
                        prev_page_text = page_texts[page_num - 1]
                        prev_page_lines = prev_page_text.splitlines()
                        # Take last 10 lines (or all if fewer) from previous page
                        overlap_lines = prev_page_lines[-10:] if len(prev_page_lines) > 10 else prev_page_lines
                        prev_overlap = "\n".join(overlap_lines)
                        chunk_text += prev_overlap + "\n\n<<<\n\n"  # Add clear page break marker
                    
                    # Add current page text
                    current_page_text = page_texts[page_num]
                    chunk_text += current_page_text
                    
                    # Add first k lines of next page if available (context after)
                    if page_num < effective_end_page and (page_num + 1) in page_texts:
                        next_page_text = page_texts[page_num + 1]
                        next_page_lines = next_page_text.splitlines()
                        # Take first 10 lines (or all if fewer) from next page
                        overlap_lines = next_page_lines[:10] if len(next_page_lines) > 10 else next_page_lines
                        next_overlap = "\n".join(overlap_lines)
                        chunk_text += "\n\n<<<\n\n" + next_overlap  # Add clear page break marker
                    
                    async with api_semaphore:
                        # Get the extraction prompt from the right client
                        if self.llm_provider == "together":
                            extraction_prompt = self.extraction_service.together_client.get_chunked_extraction_prompt(
                                chunk_text, f"chunk_{page_num}"
                            )
                        else:  # openrouter
                            extraction_prompt = self.extraction_service.openrouter_client.get_chunked_extraction_prompt(
                                chunk_text, f"chunk_{page_num}"
                            )
                        
                        # Save the extraction prompt with the OCR text included
                        nonlocal all_prompts
                        all_prompts += f"--- EXTRACTION PROMPT FOR CHUNK CENTERED ON PAGE {page_num} ---\n\n{extraction_prompt}\n\n"
                        
                        # Add a delay between requests to respect rate limits
                        await asyncio.sleep(self.delay_between_requests)
                        
                        # Process the chunk with rate limiting
                        _, json_data = await self.extraction_service.process_text_with_mistral(
                            chunk_text, f"chunk_{page_num}", is_chunk=True
                        )
                    
                    return page_num, json_data
                
                # Create tasks for all pages but with controlled concurrency
                page_tasks = []
                for page_num in pages_to_process:
                    task = asyncio.create_task(process_chunk_with_rate_limiting(page_num))
                    page_tasks.append(task)
                
                # Wait for all tasks to complete
                for task in asyncio.as_completed(page_tasks):
                    page_num, json_data = await task
                    
                    # Store the result
                    if json_data:
                        # Add page number to each question for ordering purposes
                        if "sections" in json_data:
                            for section in json_data["sections"]:
                                for question in section.get("questions", []):
                                    # Add page number metadata to preserve ordering
                                    question["_page_num"] = page_num
                        
                        combined_data = self.extraction_service.append_to_combined_data(json_data, combined_data)
                        processed_pages.append(page_num)
                        print(f"Updated combined results for '{base_name}' with page {page_num}")
                    else:
                        print(f"No structured data extracted from page {page_num} for '{base_name}'")
            else:
                # For Mistral, process pages concurrently without explicit rate limiting
                processing_tasks = []
                for page_num in pages_to_process:
                    if page_num not in page_texts:
                        continue  # Skip pages where OCR failed
                    
                    # Create a chunk containing content from adjacent pages
                    chunk_text = ""
                    
                    # Add last k lines of previous page if available (context before)
                    if page_num > start_page and (page_num - 1) in page_texts:
                        prev_page_text = page_texts[page_num - 1]
                        prev_page_lines = prev_page_text.splitlines()
                        # Take last 10 lines (or all if fewer) from previous page
                        overlap_lines = prev_page_lines[-10:] if len(prev_page_lines) > 10 else prev_page_lines
                        prev_overlap = "\n".join(overlap_lines)
                        chunk_text += prev_overlap + "\n\n<<<\n\n"  # Add clear page break marker
                    
                    # Add current page text
                    current_page_text = page_texts[page_num]
                    chunk_text += current_page_text
                    
                    # Add first k lines of next page if available (context after)
                    if page_num < effective_end_page and (page_num + 1) in page_texts:
                        next_page_text = page_texts[page_num + 1]
                        next_page_lines = next_page_text.splitlines()
                        # Take first 10 lines (or all if fewer) from next page
                        overlap_lines = next_page_lines[:10] if len(next_page_lines) > 10 else next_page_lines
                        next_overlap = "\n".join(overlap_lines)
                        chunk_text += "\n\n<<<\n\n" + next_overlap  # Add clear page break marker
                    
                    # Get the extraction prompt from the appropriate LLM client
                    extraction_prompt = self.extraction_service.mistral_client.get_chunked_extraction_prompt(
                        chunk_text, f"chunk_{page_num}"
                    )
                    
                    # Save the extraction prompt with the OCR text included
                    all_prompts += f"--- EXTRACTION PROMPT FOR CHUNK CENTERED ON PAGE {page_num} ---\n\n{extraction_prompt}\n\n"
                    
                    # Process the chunk
                    processing_task = asyncio.create_task(
                        self.extraction_service.process_text_with_mistral(
                            chunk_text, f"chunk_{page_num}", is_chunk=True
                        )
                    )
                    processing_task.set_name(f"{base_name}_process_chunk_{page_num}")
                    processing_tasks.append((page_num, processing_task))
                
                # Process results as they complete
                for page_num, processing_task in processing_tasks:
                    _, json_data = await processing_task
                    
                    if json_data:
                        # Add page number to each question for ordering purposes
                        if "sections" in json_data:
                            for section in json_data["sections"]:
                                for question in section.get("questions", []):
                                    # Add page number metadata to preserve ordering
                                    question["_page_num"] = page_num
                        
                        combined_data = self.extraction_service.append_to_combined_data(json_data, combined_data)
                        processed_pages.append(page_num)
                        print(f"Updated combined results for '{base_name}' with page {page_num}")
                    else:
                        print(f"No structured data extracted from page {page_num} for '{base_name}'")
            
            # --- Provider Identification using full document --- 
            print(f"\nIdentifying provider for '{base_name}' using voting system...")
            
            # Get the classifier prompt from the appropriate LLM client
            if self.llm_provider == "together":
                classifier_prompt = self.provider_service.together_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            elif self.llm_provider == "openrouter":
                classifier_prompt = self.provider_service.openrouter_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            elif self.llm_provider == "deepseek":
                classifier_prompt = self.provider_service.deepseek_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            elif self.llm_provider == "claude":
                classifier_prompt = self.provider_service.claude_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            elif self.llm_provider == "gemini":
                classifier_prompt = self.provider_service.gemini_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            else:
                classifier_prompt = self.provider_service.mistral_client.get_classifier_prompt(
                    all_ocr_text, file_extension
                )
            
            # Save the classifier prompt
            all_prompts += f"\n\n--- PROVIDER CLASSIFICATION PROMPT ---\n\n{classifier_prompt}\n"
            
            # Save all prompts to output/prompts/pdfname.txt
            await write_file_async(prompts_output_path, all_prompts)
            
            provider = await self.provider_service.identify_provider(all_ocr_text, file_extension)
            print(f"Identified provider for '{base_name}': {provider.value}")
            
            # Convert to mutation format using the identified provider
            mutation_data = self.extraction_service.convert_to_mutation_format(combined_data, provider)
            
            # Final save of structured and mutation data
            await write_file_async(structured_output_path, json.dumps(combined_data, indent=2))
            await write_file_async(mutation_output_path, json.dumps(mutation_data, indent=2))
            print(f"Saved final results for '{base_name}' to {structured_output_path} and {mutation_output_path}")
            print(f"Saved OCR text to {ocr_output_path}")
            print(f"Saved prompts to {prompts_output_path}")
            
            # Clean up the run-specific temp directory
            try:
                await rmtree_async(run_temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory {run_temp_dir}: {e}")
            
            processed_pages.sort()
            
            # Return the final data and info
            return (
                str(structured_output_path), 
                str(mutation_output_path), 
                json.dumps(combined_data, indent=2),  # Return the JSON string itself
                processed_pages, 
                provider
            ) 