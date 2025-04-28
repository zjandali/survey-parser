#!/usr/bin/env python3
"""
Batch PDF Survey Processor

This script processes PDF files containing surveys, extracts structured data,
and outputs JSON files in mutation format.

Usage:
  python main.py input_directory [-c concurrency] [-t timeout] [-p pattern] [--max-ram-percent percent] [--llm-provider provider] [--debug-prompts] [--output-dir directory]

Positional arguments:
  input_directory     Directory containing PDF files to process

Optional arguments:
  -c, --concurrency    Maximum number of files to process simultaneously (default: provider-specific)
  -t, --timeout        Timeout in seconds for OCR processing (default: 300)
  -p, --pattern        File pattern to match (default: "*.pdf,*.doc,*.docx")
  --max-ram-percent    Maximum RAM usage percent threshold (default: 80)
  --llm-provider       LLM provider to use: 'mistral', 'together', 'openrouter', 'deepseek', 'claude', or 'gemini' (default: mistral)
  --debug-prompts      Output all LLM prompts to terminal
  --output-dir         Directory to save output files (default: "output")
"""

import sys
import argparse
import asyncio
from pathlib import Path
import os

from processors.batch_processor import BatchProcessor
from config.settings import (
    DEFAULT_TIMEOUT,
    DEFAULT_FILE_PATTERN,
    DEFAULT_MAX_RAM_PERCENT,
    DEFAULT_LLM_PROVIDER,
    TOGETHER_CONCURRENCY_LIMIT,
    MISTRAL_CONCURRENCY_LIMIT,
    OPENROUTER_CONCURRENCY_LIMIT,
    DEEPSEEK_CONCURRENCY_LIMIT,
    CLAUDE_CONCURRENCY_LIMIT,
    GEMINI_CONCURRENCY_LIMIT
)

async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Batch process multiple PDF files with async survey parser")
    parser.add_argument("input_dir", 
                        help="Directory containing PDF files to process")
    parser.add_argument("-c", "--concurrency", type=int,
                        help="Maximum number of files to process simultaneously (default: provider-specific)")
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT, 
                        help=f"Timeout in seconds for OCR processing (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("-p", "--pattern", default="*.pdf,*.doc,*.docx", 
                        help=f"File pattern to match (default: '*.pdf,*.doc,*.docx')")
    parser.add_argument("--max-ram-percent", type=float, default=DEFAULT_MAX_RAM_PERCENT, 
                        help=f"Maximum RAM usage percent threshold (default: {DEFAULT_MAX_RAM_PERCENT})")
    parser.add_argument("--llm-provider", choices=['mistral', 'together', 'openrouter', 'deepseek', 'claude', 'gemini'], default=DEFAULT_LLM_PROVIDER,
                        help=f"LLM provider to use: 'mistral', 'together', 'openrouter', 'deepseek', 'claude', or 'gemini' (default: {DEFAULT_LLM_PROVIDER})")
    parser.add_argument("--debug-prompts", action="store_true",
                        help="Output all LLM prompts to terminal")
    parser.add_argument("--output-dir", default="output", 
                        help="Directory to save output files (default: 'output')")
    
    args = parser.parse_args()
    
    try:
        # Set environment variable for debugging prompts
        if args.debug_prompts:
            os.environ["DEBUG_LLM_PROMPTS"] = "1"
            print("Debug mode enabled: All LLM prompts will be printed to terminal")
        
        batch_processor = BatchProcessor(llm_provider=args.llm_provider)
        
        # Print appropriate warnings/information based on provider
        if args.llm_provider == "together" and args.concurrency and args.concurrency > TOGETHER_CONCURRENCY_LIMIT:
            provider_specific_limit = TOGETHER_CONCURRENCY_LIMIT
            print(f"Warning: Together API has a rate limit of 600 RPM (10 RPS) on Tier 1.")
            print(f"Limiting concurrency from {args.concurrency} to {provider_specific_limit} to avoid rate limiting.")
            print(f"Pages will still be processed in parallel with controlled rate limiting.")
        elif args.llm_provider == "together":
            print(f"Note: Using Together API with concurrency {TOGETHER_CONCURRENCY_LIMIT}.")
            print(f"Pages will be processed in parallel with controlled rate limiting to respect the 600 RPM Tier 1 limit.")
        elif args.llm_provider == "openrouter" and args.concurrency and args.concurrency > OPENROUTER_CONCURRENCY_LIMIT:
            provider_specific_limit = OPENROUTER_CONCURRENCY_LIMIT
            print(f"Warning: Using OpenRouter with conservative rate limits.")
            print(f"Limiting concurrency from {args.concurrency} to {provider_specific_limit}.")
            print(f"Pages will be processed in parallel with controlled rate limiting.")
        elif args.llm_provider == "openrouter":
            print(f"Note: Using OpenRouter API with DeepSeek model.")
            print(f"Using concurrency {OPENROUTER_CONCURRENCY_LIMIT} with controlled rate limiting.")
        elif args.llm_provider == "deepseek" and args.concurrency and args.concurrency > DEEPSEEK_CONCURRENCY_LIMIT:
            provider_specific_limit = DEEPSEEK_CONCURRENCY_LIMIT
            print(f"Warning: Using DeepSeek R1 via OpenRouter with conservative rate limits.")
            print(f"Limiting concurrency from {args.concurrency} to {provider_specific_limit}.")
            print(f"Pages will be processed in parallel with controlled rate limiting.")
        elif args.llm_provider == "deepseek":
            print(f"Note: Using DeepSeek R1 model via OpenRouter.")
            print(f"Using concurrency {DEEPSEEK_CONCURRENCY_LIMIT} with controlled rate limiting.")
        elif args.llm_provider == "claude" and args.concurrency and args.concurrency > CLAUDE_CONCURRENCY_LIMIT:
            provider_specific_limit = CLAUDE_CONCURRENCY_LIMIT
            print(f"Warning: Using Claude-3.7-Sonnet via OpenRouter with conservative rate limits.")
            print(f"Limiting concurrency from {args.concurrency} to {provider_specific_limit}.")
            print(f"Pages will be processed in parallel with controlled rate limiting.")
        elif args.llm_provider == "claude":
            print(f"Note: Using Claude-3.7-Sonnet model via OpenRouter.")
            print(f"Using concurrency {CLAUDE_CONCURRENCY_LIMIT} with controlled rate limiting.")
        elif args.llm_provider == "gemini" and args.concurrency and args.concurrency > GEMINI_CONCURRENCY_LIMIT:
            provider_specific_limit = GEMINI_CONCURRENCY_LIMIT
            print(f"Warning: Using Gemini-2.5-Pro via OpenRouter with conservative rate limits.")
            print(f"Limiting concurrency from {args.concurrency} to {provider_specific_limit}.")
            print(f"Pages will be processed in parallel with controlled rate limiting.")
        elif args.llm_provider == "gemini":
            print(f"Note: Using Gemini-2.5-Pro model via OpenRouter.")
            print(f"Using concurrency {GEMINI_CONCURRENCY_LIMIT} with controlled rate limiting.")
        
        results = await batch_processor.process_directory(
            args.input_dir,
            concurrency=args.concurrency,  # Will use provider-specific default if None
            timeout=args.timeout,
            pattern=args.pattern,
            max_ram_percent=args.max_ram_percent,

        )
        
        # Return 0 for success, 1 for error
        if "error" in results:
            return 1
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        return 1
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Entry point to run the async batch processor"""
    # Set recursion limit higher for JSON processing
    sys.setrecursionlimit(10000)
    
    # Run the async main function
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)

if __name__ == "__main__":
    import shutil, os
    # Get arguments for output dir before running main
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default="output")
    args, _ = parser.parse_known_args()
    
    # Clean output directory if it exists
    if os.path.exists(args.output_dir): 
        shutil.rmtree(args.output_dir)

    main()