"""
Service for survey provider identification.
"""

import asyncio
import logging
import os
from typing import Optional

from api.mistral_client import MistralClient
from api.openrouter_client import OpenRouterClient
from api.deepseek_client import DeepSeekClient
from api.claude_client import ClaudeClient
from api.gemini_client import GeminiClient
from models.enums import FeasibilityProvider
from core.text_utils import chunk_text

class ProviderService:
    """Service for survey provider identification"""
    
    def __init__(self, llm_provider: str = "mistral"):
        """
        Initialize the provider service
        
        Args:
            llm_provider: LLM provider to use ("mistral",  "openrouter", "deepseek", "claude", or "gemini")
        """
        self.mistral_client = MistralClient()
        self.openrouter_client = OpenRouterClient()
        self.deepseek_client = DeepSeekClient()
        self.claude_client = ClaudeClient()
        self.gemini_client = GeminiClient()
        self.llm_provider = llm_provider
        
        # Debug mode for printing prompts to terminal
        self.debug_prompts = os.environ.get("DEBUG_LLM_PROMPTS", "0") == "1"
        
        # Rate limiting parameters - adjust based on provider's API limits
        if llm_provider == "together":
            self.concurrency_limit = 1
            self.delay_between_requests = 1.5
        elif llm_provider == "openrouter":
            self.concurrency_limit = 1
            self.delay_between_requests = 2.0  # More conservative for OpenRouter
        elif llm_provider == "deepseek":
            self.concurrency_limit = 1
            self.delay_between_requests = 2.0  # Same conservative approach for DeepSeek
        elif llm_provider == "claude":
            self.concurrency_limit = 1
            self.delay_between_requests = 2.0  # Same conservative approach for Claude
        elif llm_provider == "gemini":
            self.concurrency_limit = 1
            self.delay_between_requests = 2.0  # Same conservative approach for Gemini
        else:  # mistral
            self.concurrency_limit = 5
            self.delay_between_requests = 0
    
    async def process_chunk(
        self, 
        chunk_content: str, 
        chunk_position: int, 
        chunk_number: int, 
        total_chunks: int, 
        file_extension: str
    ) -> Optional[FeasibilityProvider]:
        """
        Process a single text chunk to identify the provider
        
        Args:
            chunk_content: Chunk text content
            chunk_position: Start position of chunk
            chunk_number: Chunk number
            total_chunks: Total number of chunks
            file_extension: File extension
            
        Returns:
            Identified provider or None if classification failed
        """
        retry_count = 0
        retry_delay = 2.0
        
        while True:  # Infinite retry loop
            try:
                # Create chunk info
                chunk_info = f"Chunk {chunk_number} of {total_chunks} (starting at position {chunk_position})"
                formatted_prompt = None
                
                # Format the prompt using the appropriate client
                if self.llm_provider == "together":
                    formatted_prompt = self.together_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nTOGETHER CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.together_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "openrouter":
                    formatted_prompt = self.openrouter_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nOPENROUTER CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.openrouter_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "deepseek":
                    formatted_prompt = self.deepseek_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nDEEPSEEK CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.deepseek_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "claude":
                    formatted_prompt = self.claude_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nCLAUDE CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.claude_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "gemini":
                    formatted_prompt = self.gemini_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nGEMINI CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.gemini_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                else:
                    formatted_prompt = self.mistral_client.get_classifier_prompt(
                        chunk_content, file_extension, is_chunk=True, chunk_info=chunk_info
                    )
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nMISTRAL CHUNK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.mistral_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                
                # Extract just the provider enum value from the response
                raw_result = result.strip()
                if self.llm_provider == "gemini":
                    logging.info(f"Gemini provider classification raw response: '{raw_result}'")
                provider_name = self._extract_provider_enum(raw_result)
                
                if not provider_name and not raw_result and self.llm_provider == "gemini":
                    # For empty Gemini responses, use the default based on file extension
                    logging.warning(f"Empty response from Gemini, using default provider for {file_extension}")
                    provider_name = self._get_default_provider(file_extension).value
                
                if provider_name:
                    print(f"{self.llm_provider.capitalize()} classified chunk {chunk_number}/{total_chunks} as: {provider_name}")
                    
                    # Convert to enum
                    try:
                        return FeasibilityProvider(provider_name)
                    except ValueError:
                        print(f"Invalid provider classification for chunk {chunk_number}: {provider_name}")
                        if retry_count < 5:  # Limit retries to 5
                            retry_count += 1
                            print(f"Retrying chunk {chunk_number} classification (attempt {retry_count}/5)...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            # After 5 retries, fall back to a default based on file extension
                            print(f"Maximum retries reached for chunk {chunk_number}, falling back to file extension")
                            return self._get_default_provider(file_extension)
                else:
                    print(f"Could not extract valid provider from response: {raw_result[:100]}...")
                    if retry_count < 5:
                        retry_count += 1
                        print(f"Retrying chunk {chunk_number} classification (attempt {retry_count}/5)...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # After 5 retries, fall back to a default based on file extension
                        print(f"Maximum retries reached for chunk {chunk_number}, falling back to file extension")
                        return self._get_default_provider(file_extension)
                    
            except Exception as e:
                print(f"Error classifying chunk {chunk_number}: {str(e)}")
                if retry_count < 5:
                    retry_count += 1
                    print(f"Retrying chunk {chunk_number} classification (attempt {retry_count}/5)...")
                    await asyncio.sleep(retry_delay * (1.5 ** min(retry_count, 5)))  # Exponential backoff with cap
                    continue
                else:
                    # After 5 retries, fall back to a default based on file extension
                    print(f"Maximum retries reached for chunk {chunk_number}, falling back to file extension")
                    return self._get_default_provider(file_extension)
                
        # This should not be reached with the continue statements
        return self._get_default_provider(file_extension)
    
    def _extract_provider_enum(self, text: str) -> Optional[str]:
        """
        Extract a valid provider enum value from text
        
        Args:
            text: Text to extract from
            
        Returns:
            Valid enum value or None
        """
        if not text or not text.strip():
            return None
            
        # List of valid enum values
        valid_values = [e.value for e in FeasibilityProvider]
        
        # First, check if the text exactly matches a valid enum value
        if text in valid_values:
            return text
            
        # Remove any line breaks and extra whitespace for better matching
        cleaned_text = ' '.join(text.strip().split())
        
        # Check for direct match after cleaning
        if cleaned_text in valid_values:
            return cleaned_text
            
        # Some models might return "Microsoft Forms" instead of "MICROSOFT_FORMS"
        normalized_text = cleaned_text.upper().replace(' ', '_')
        if normalized_text in valid_values:
            return normalized_text
            
        # If the model returns something like "The survey provider is MICROSOFT_FORMS"
        # try to extract just the enum value
        for value in valid_values:
            # Direct substring match (highest confidence)
            if value in cleaned_text:
                return value
                
            # Try to match even with lowercase/mixed case
            if value.lower() in cleaned_text.lower():
                return value
                
        # For longer explanatory texts (especially from Gemini),
        # look for keywords that strongly indicate specific providers
        lower_text = cleaned_text.lower()
        
        # Keyword-based fallback matching
        if any(kw in lower_text for kw in ["microsoft forms", "forms.office.com", "office form"]):
            return FeasibilityProvider.MICROSOFT_FORMS.value
        elif any(kw in lower_text for kw in ["surveymonkey", "survey monkey"]):
            return FeasibilityProvider.SURVEY_MONKEY.value
        elif any(kw in lower_text for kw in ["google forms", "google form", "docs.google.com/forms"]):
            return FeasibilityProvider.GOOGLE_FORMS.value
        elif any(kw in lower_text for kw in ["qualtrics"]):
            return FeasibilityProvider.QUALTRICS.value
        elif any(kw in lower_text for kw in ["pdf document", "pdf file"]):
            return FeasibilityProvider.PDF.value
        elif any(kw in lower_text for kw in ["microsoft word", "word document", "doc file", "docx"]):
            return FeasibilityProvider.MICROSOFT_WORD.value
        
        # No valid match found
        return None
    
    def _get_default_provider(self, file_extension: str) -> FeasibilityProvider:
        """
        Get a default provider based on file extension
        
        Args:
            file_extension: File extension
            
        Returns:
            Default provider
        """
        if file_extension.lower() == 'pdf':
            return FeasibilityProvider.PDF
        elif file_extension.lower() in ['doc', 'docx']:
            return FeasibilityProvider.MICROSOFT_WORD
        else:
            return FeasibilityProvider.NULL
    
    def determine_provider_from_extension(self, file_extension: str) -> FeasibilityProvider:
        """
        Determine the provider based on file extension alone
        
        Args:
            file_extension: File extension
            
        Returns:
            The determined provider
        """
        return self._get_default_provider(file_extension)
    
    async def identify_provider(self, ocr_text: str, file_extension: str) -> FeasibilityProvider:
        """
        Use LLM to analyze OCR text and identify the provider using chunk-based voting
        
        Args:
            ocr_text: OCR text to analyze
            file_extension: File extension
            
        Returns:
            The identified provider
        """
        if not ocr_text:
            print("No OCR text provided for provider identification. Falling back to file extension.")
            return self._get_default_provider(file_extension)
        
        # Process the entire OCR text as a whole first
        retry_count = 0
        max_retries = 3  # Limit retries for full document analysis
        retry_delay = 2.0
        
        full_document_provider = None
        
        # First try to process the entire document
        while retry_count <= max_retries:
            try:
                formatted_prompt = None
                
                if self.llm_provider == "together":
                    formatted_prompt = self.together_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nTOGETHER CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.together_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "openrouter":
                    formatted_prompt = self.openrouter_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nOPENROUTER CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.openrouter_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "deepseek":
                    formatted_prompt = self.deepseek_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nDEEPSEEK CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.deepseek_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "claude":
                    formatted_prompt = self.claude_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nCLAUDE CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.claude_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "gemini":
                    formatted_prompt = self.gemini_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nGEMINI CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.gemini_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                else:
                    formatted_prompt = self.mistral_client.get_classifier_prompt(ocr_text, file_extension)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nMISTRAL CLASSIFICATION PROMPT\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.mistral_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                
                # Extract just the provider enum value from the response
                raw_result = result.strip()
                if self.llm_provider == "gemini":
                    logging.info(f"Gemini provider classification raw response: '{raw_result}'")
                provider_name = self._extract_provider_enum(raw_result)
                
                if not provider_name and not raw_result and self.llm_provider == "gemini":
                    # For empty Gemini responses, use the default based on file extension
                    logging.warning(f"Empty response from Gemini, using default provider for {file_extension}")
                    provider_name = self._get_default_provider(file_extension).value
                
                if provider_name:
                    print(f"{self.llm_provider.capitalize()} classified the whole document as: {provider_name}")
                    
                    # Convert to enum
                    try:
                        full_document_provider = FeasibilityProvider(provider_name)
                        # Non-default providers from full document analysis are very reliable
                        # Return them immediately if they're not just file type defaults
                        if full_document_provider not in [FeasibilityProvider.PDF, FeasibilityProvider.MICROSOFT_WORD, FeasibilityProvider.NULL]:
                            return full_document_provider
                        break  # We got a valid result, but it's a default, so proceed to chunking as a verification
                    except ValueError:
                        print(f"Invalid provider classification: {provider_name}")
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"Retrying document classification (attempt {retry_count}/{max_retries})...")
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.2, 10)
                            continue
                else:
                    print(f"Could not extract valid provider from full document response: {raw_result[:100]}...")
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(f"Retrying document classification (attempt {retry_count}/{max_retries})...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.2, 10)
                        continue
                    
            except Exception as e:
                print(f"Error identifying provider with {self.llm_provider.capitalize()}: {str(e)}")
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"Retrying document classification (attempt {retry_count}/{max_retries})...")
                    await asyncio.sleep(retry_delay * (1.5 ** min(retry_count, 3)))
                    continue
                
        # As a fallback, use a chunking approach with larger chunks
        # Split text into manageable chunks, but with larger size and minimal chunks
        CHUNK_SIZE = 8000  # Large chunk size to reduce number of chunks
        OVERLAP = 500
        
        chunks = chunk_text(ocr_text, CHUNK_SIZE, OVERLAP)
        total_chunks = len(chunks)
        print(f"Splitting document into {total_chunks} chunks for provider voting analysis")
        
        # Use a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        async def process_chunk_with_limit(chunk_content, chunk_position, chunk_number, total_chunks, file_extension):
            async with semaphore:
                # Add a delay between requests to respect rate limits
                if self.delay_between_requests > 0 and chunk_number > 1:
                    await asyncio.sleep(self.delay_between_requests)
                return await self.process_chunk(chunk_content, chunk_position, chunk_number, total_chunks, file_extension)
        
        # Process chunks with limited concurrency
        tasks = []
        for i, (chunk_content, chunk_position) in enumerate(chunks):
            task = asyncio.create_task(
                process_chunk_with_limit(chunk_content, chunk_position, i + 1, total_chunks, file_extension)
            )
            task.set_name(f"provider_chunk_{i+1}")
            tasks.append(task)
        
        chunk_results = []
        for task in tasks:
            provider = await task
            if provider:
                chunk_results.append(provider)
        
        if not chunk_results:
            print("No valid provider classifications from chunks.")
            if full_document_provider:
                print(f"Using full document classification: {full_document_provider.value}")
                return full_document_provider
            print("Falling back to file extension.")
            return self._get_default_provider(file_extension)
        
        # Find the most common result using a sophisticated voting scheme
        from collections import Counter
        vote_counter = Counter(chunk_results)
        
        # Print voting results for debug
        print(f"Provider voting results: {[(p.value, count) for p, count in vote_counter.items()]}")
        
        # Prioritize specific providers over file-based defaults
        specific_providers = [p for p in vote_counter.keys() 
                             if p not in [FeasibilityProvider.PDF, FeasibilityProvider.MICROSOFT_WORD, FeasibilityProvider.NULL]]
        
        if specific_providers:
            specific_counts = {p: vote_counter[p] for p in specific_providers}
            # Choose the most frequent specific provider
            most_common_specific = max(specific_counts, key=specific_counts.get)
            
            # Get total vote count and vote count for the most common specific provider
            total_votes = sum(vote_counter.values())
            specific_vote_count = specific_counts[most_common_specific]
            specific_vote_percentage = (specific_vote_count / total_votes) * 100
            
            # Check if specific provider has enough votes (at least 30% of total)
            if specific_vote_percentage >= 30:
                print(f"Selected specific provider based on {specific_vote_count} votes ({specific_vote_percentage:.1f}% of total): {most_common_specific.value}")
                return most_common_specific
            else:
                print(f"Specific provider {most_common_specific.value} only has {specific_vote_count} votes ({specific_vote_percentage:.1f}% of total), which is below the 30% threshold")
                # Fall through to check other options rather than returning immediately
        
        # Find the most common overall provider (including file-based defaults)
        most_common = vote_counter.most_common(1)[0][0]
        most_common_count = vote_counter[most_common]
        
        # If we have a full document classification that matches the most common vote, use it
        if full_document_provider and full_document_provider == most_common:
            print(f"Using full document classification which matches most common vote: {full_document_provider.value} ({most_common_count} votes)")
            return full_document_provider
        # Otherwise if we have a full document classification and no specific provider met the threshold, use it
        elif full_document_provider:
            print(f"No specific provider met the threshold. Using full document classification: {full_document_provider.value}")
            return full_document_provider
            
        # Lastly, fallback to the most common overall vote
        print(f"Selected provider based on most frequent vote ({most_common_count} votes): {most_common.value}")
        return most_common 