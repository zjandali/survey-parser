"""
Service for extracting structured data from OCR text.
"""

import json
import uuid
import asyncio
import logging
import re
import os
from typing import Dict, Any, List, Tuple, Optional

from api.mistral_client import MistralClient
from api.together_client import TogetherClient
from api.openrouter_client import OpenRouterClient
from api.deepseek_client import DeepSeekClient
from api.claude_client import ClaudeClient
from api.gemini_client import GeminiClient
from models.enums import FeasibilityProvider

class ExtractionService:
    """Service for structured data extraction from OCR text"""
    
    def __init__(self, llm_provider: str = "mistral"):
        """
        Initialize the extraction service
        
        Args:
            llm_provider: LLM provider to use ("mistral", "together", "openrouter", "deepseek", "claude", or "gemini")
        """
        self.mistral_client = MistralClient()
        self.together_client = TogetherClient()
        self.openrouter_client = OpenRouterClient()
        self.deepseek_client = DeepSeekClient()
        self.claude_client = ClaudeClient()
        self.gemini_client = GeminiClient()
        self.llm_provider = llm_provider
        
        # Rate limiting parameters - adjust based on provider
        if llm_provider == "together":
            self.delay_between_retries = 2.0
        elif llm_provider in ["openrouter", "deepseek", "claude", "gemini"]:
            self.delay_between_retries = 3.0  # More conservative for external providers
        else:  # mistral
            self.delay_between_retries = 1.0
        
        # Debug mode for printing prompts to terminal
        self.debug_prompts = os.environ.get("DEBUG_LLM_PROMPTS", "0") == "1"
        
        # Common section headings that might trigger hallucinations
        self.risk_section_headings = [
            "clinical raters", "rating scales", "assessments", 
            "cognitive assessments", "neuropsychological assessments",
            "qualification", "certified raters"
        ]
    
    def validate_and_filter_hallucinations(self, json_data: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
        """
        Validate JSON data against original OCR text to detect potential hallucinations
        
        Args:
            json_data: Parsed JSON data from LLM
            ocr_text: Original OCR text that was processed
            
        Returns:
            Filtered JSON data with potential hallucinations removed
        """
        if not json_data or "sections" not in json_data:
            return json_data
            
        # First, check if any risk section headings appear near the end of the document
        # and if they do, flag those sections for extra scrutiny
        risk_headings_at_end = []
        last_n_chars = ocr_text[-1000:].lower()  # Check last 1000 characters
        
        for heading in self.risk_section_headings:
            if heading.lower() in last_n_chars:
                position = last_n_chars.find(heading.lower())
                if position != -1 and position > len(last_n_chars) - 200:  # If very near the end
                    risk_headings_at_end.append(heading.lower())
                    logging.warning(f"Detected risk heading '{heading}' near document end. Applying extra validation.")
        
        # If no risk headings at end, return original data
        if not risk_headings_at_end:
            return json_data
            
        # Look for questions mentioning common clinical rating scales
        common_scales = [
            "mmse", "cdr", "adas[-\\s]?cog", "adcs[-\\s]?adl", "npi", "moca", 
            "gds", "c[-\\s]?ssrs", "updrs", "faq", "qol[-\\s]?ad"
        ]
        
        filtered_sections = []
        for section in json_data["sections"]:
            section_title = section.get("title", "").lower()
            
            # If this section has a title matching or similar to a risk heading
            if any(heading in section_title for heading in risk_headings_at_end):
                # Extra validation for questions in this section
                if "questions" in section:
                    filtered_questions = []
                    for question in section["questions"]:
                        # Check if the question mentions any common clinical scales
                        question_text = question.get("text", "").lower()
                        matches_scale = any(re.search(scale, question_text, re.IGNORECASE) for scale in common_scales)
                        
                        # Also check if the question or any options mention "yes" as the only option
                        has_just_yes = False
                        if question.get("selected") == "Yes" or question.get("selected") == ["Yes"]:
                            options = question.get("options", [])
                            if len(options) == 1 and options[0].lower() == "yes":
                                has_just_yes = True
                        
                        # If suspicious question, exclude it
                        if matches_scale or has_just_yes:
                            logging.warning(f"Filtered out likely hallucinated question: {question_text}")
                            continue
                        
                        filtered_questions.append(question)
                    
                    # Update the section with filtered questions
                    section["questions"] = filtered_questions
            
            filtered_sections.append(section)
        
        json_data["sections"] = filtered_sections
        return json_data
    
    async def process_text_with_mistral(
        self, 
        ocr_text: str, 
        page_info: str,
        is_chunk: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process OCR text with the LLM to extract structured data
        
        Args:
            ocr_text: OCR text to process
            page_info: Page identifier (for logging)
            is_chunk: Whether this is a chunk of a larger text
            
        Returns:
            Tuple of (result_text, structured_data)
        """
        if not ocr_text or not ocr_text.strip():
            return None, None
            
        print(f"Processing text with {self.llm_provider.capitalize()}: {page_info}")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Format the prompt based on the selected provider
                if self.llm_provider == "together":
                    if is_chunk:
                        formatted_prompt = self.together_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.together_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nTOGETHER PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.together_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                    # For Together, the client already does JSON cleaning and formatting
                    # so we should have valid JSON here
                elif self.llm_provider == "openrouter":
                    if is_chunk:
                        formatted_prompt = self.openrouter_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.openrouter_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nOPENROUTER PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.openrouter_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                    # For OpenRouter, the client already does JSON cleaning and formatting
                    # so we should have valid JSON here
                elif self.llm_provider == "deepseek":
                    if is_chunk:
                        formatted_prompt = self.deepseek_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.deepseek_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nDEEPSEEK PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.deepseek_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "claude":
                    if is_chunk:
                        formatted_prompt = self.claude_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.claude_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nCLAUDE PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.claude_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                elif self.llm_provider == "gemini":
                    if is_chunk:
                        formatted_prompt = self.gemini_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.gemini_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nGEMINI PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.gemini_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                else:  # mistral
                    if is_chunk:
                        formatted_prompt = self.mistral_client.get_chunked_extraction_prompt(ocr_text, page_info)
                    else:
                        formatted_prompt = self.mistral_client.get_extraction_prompt(ocr_text, page_info)
                    # Debug print if enabled
                    if self.debug_prompts:
                        print(f"\n{'='*80}\nMISTRAL PROMPT FOR {page_info}\n{'='*80}\n{formatted_prompt}\n{'='*80}\n")
                    result = await self.mistral_client.complete_async(
                        prompt=formatted_prompt,
                        max_tokens=100000,  # No token limit
                        temperature=0.1
                    )
                    
                cleaned_result = result.strip()
                
                # Parse JSON response
                try:
                    # First, try direct parsing
                    json_data = json.loads(cleaned_result)
                    print(f"Successfully parsed and processed data for {page_info}")
                    return cleaned_result, json_data
                except json.JSONDecodeError as e:
                    # Try to extract JSON from the text - this handles cases with markdown formatting or extra text
                    try:
                        # Look for JSON-like content between triple backticks
                        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                        match = re.search(json_pattern, cleaned_result)
                        if match:
                            json_str = match.group(1).strip()
                            json_data = json.loads(json_str)
                            print(f"Successfully extracted and parsed JSON from markdown for {page_info}")
                            return cleaned_result, json_data
                        else:
                            # Look for content that appears to be JSON (starts with { and ends with })
                            json_pattern = r"(\{[\s\S]*\})"
                            match = re.search(json_pattern, cleaned_result)
                            if match:
                                json_str = match.group(1).strip()
                                json_data = json.loads(json_str)
                                print(f"Successfully extracted and parsed JSON from text for {page_info}")
                                return cleaned_result, json_data
                            
                            # If we got here, we couldn't parse the JSON
                            if attempt < max_attempts - 1:
                                print(f"Failed to parse JSON for {page_info} (attempt {attempt+1}/{max_attempts}). Retrying...")
                                await asyncio.sleep(self.delay_between_retries * (attempt + 1))
                                continue
                            else:
                                print(f"Failed to parse JSON for {page_info} after {max_attempts} attempts")
                                return cleaned_result, None
                    except Exception as inner_e:
                        if attempt < max_attempts - 1:
                            print(f"Error processing JSON for {page_info} (attempt {attempt+1}/{max_attempts}): {str(inner_e)}. Retrying...")
                            await asyncio.sleep(self.delay_between_retries * (attempt + 1))
                            continue
                        else:
                            print(f"Error processing JSON for {page_info} after {max_attempts} attempts: {str(inner_e)}")
                            return cleaned_result, None
            
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Error processing {page_info} with {self.llm_provider} (attempt {attempt+1}/{max_attempts}): {str(e)}. Retrying...")
                    await asyncio.sleep(self.delay_between_retries * (attempt + 1))
                    continue
                else:
                    print(f"Error processing {page_info} with {self.llm_provider} after {max_attempts} attempts: {str(e)}")
                    return None, None 
            
        # Should not reach here, but just in case
        return None, None
    
    def append_to_combined_data(self, json_data: Dict[str, Any], combined_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append new data to the combined data structure, handling duplicates and incompletes
        
        Args:
            json_data: New data to append
            combined_data: Existing combined data
            
        Returns:
            Updated combined data
        """
        if not json_data:
            return combined_data
        
        if not combined_data:
            combined_data = {"sections": []}
        
        section_map = {}
        question_map = {}
        
        # Build index of existing sections and questions
        for section_idx, section in enumerate(combined_data["sections"]):
            section_title = section.get("title", "Unnamed Section")
            section_map[section_title] = section_idx
            
            for question_idx, question in enumerate(section.get("questions", [])):
                # Create a more specific identifier that includes section, ID, and text
                question_id = question.get("id", "")
                question_text = question.get("text", "").strip()
                section_title = section.get("title", "Unnamed Section")
                
                # Create a composite key that better identifies distinct questions
                # Format: section:id:text
                question_key = f"{section_title}:{question_id}:{question_text}"
                
                # For common fields like Email, First Name, Last Name that might appear multiple times,
                # we need to include context - use the section's position as additional context
                if question_text.lower() in ["email:", "email", "first name:", "first name", 
                                             "last name:", "last name"]:
                    question_key = f"{section_idx}:{question_key}"
                
                if question_key:
                    question_map[question_key] = (section_idx, question_idx)
        
        # Process each section in the new data
        for section in json_data.get("sections", []):
            section_title = section.get("title", "Unnamed Section")
            
            # Find or create the section
            if section_title in section_map:
                section_idx = section_map[section_title]
            else:
                section_idx = len(combined_data["sections"])
                section_map[section_title] = section_idx
                combined_data["sections"].append({
                    "title": section_title,
                    "questions": []
                })
            
            # Process each question in the section
            for question in section.get("questions", []):
                # Create the same composite key for the new question
                question_id = question.get("id", "")
                question_text = question.get("text", "").strip()
                
                # Same composite key format for the new question
                question_key = f"{section_title}:{question_id}:{question_text}"
                
                # Apply the same special handling for common fields
                if question_text.lower() in ["email:", "email", "first name:", "first name", 
                                             "last name:", "last name"]:
                    question_key = f"{section_idx}:{question_key}"
                
                if not question_text:
                    continue
                
                # Make sure we have a valid options array
                if "options" not in question:
                    question["options"] = []
                # Ensure options is never None
                if question["options"] is None:
                    question["options"] = []
                    
                # Normalize selected field - ensure it's consistently either a string or an array
                if "selected" in question:
                    if question["selected"] is None:
                        question["selected"] = []
                    elif not isinstance(question["selected"], list):
                        question["selected"] = [question["selected"]]
                
                # Check if this question exists using the composite key
                if question_key in question_map:
                    existing_section_idx, existing_question_idx = question_map[question_key]
                    existing_question = combined_data["sections"][existing_section_idx]["questions"][existing_question_idx]
                    
                    # Normalize the existing question's selected field too
                    if "selected" in existing_question:
                        if existing_question["selected"] is None:
                            existing_question["selected"] = []
                        elif not isinstance(existing_question["selected"], list):
                            existing_question["selected"] = [existing_question["selected"]]
                    else:
                        existing_question["selected"] = []
                    
                    # Handle incomplete questions
                    if "[INCOMPLETE]" in existing_question.get("text", "") and "[INCOMPLETE]" not in question_text:
                        # Transfer page number info if necessary
                        if "_page_num" in question and "_page_num" not in existing_question:
                            question["_page_num"] = existing_question.get("_page_num")
                        
                        # Preserve any existing selected answers when replacing the question
                        if existing_question.get("selected") and not question.get("selected"):
                            question["selected"] = existing_question.get("selected", [])
                        elif existing_question.get("selected") and question.get("selected"):
                            # Merge both selected answers
                            merged_answers = []
                            for ans in existing_question.get("selected", []):
                                if ans and ans not in merged_answers:
                                    merged_answers.append(ans)
                            for ans in question.get("selected", []):
                                if ans and ans not in merged_answers:
                                    merged_answers.append(ans)
                            question["selected"] = merged_answers
                        
                        # Preserve options if the new question doesn't have them
                        if not question.get("options") and existing_question.get("options"):
                            question["options"] = existing_question.get("options", [])
                        
                        # Replace the incomplete question with the complete one
                        combined_data["sections"][existing_section_idx]["questions"][existing_question_idx] = question
                    else:
                        # For complete questions or both incomplete, we need to merge data appropriately
                        
                        # Always merge options to get the complete set
                        all_options = []
                        if existing_question.get("options") is not None:
                            all_options = list(existing_question.get("options", []))
                        else:
                            existing_question["options"] = []
                        
                        new_options = question.get("options", [])
                        if new_options is not None:
                            for opt in new_options:
                                if opt not in all_options:
                                    all_options.append(opt)
                        existing_question["options"] = all_options
                        
                        # Merge answers (the selected field)
                        if not existing_question.get("selected") and question.get("selected"):
                            # If existing has no answers but new one does, use the new answers
                            existing_question["selected"] = question.get("selected", [])
                        elif question.get("selected"):
                            # Prioritize the answer(s) from the new chunk, overwrite the old.
                            existing_question["selected"] = question.get("selected", [])
                else:
                    # This is a new question - add it to the appropriate section
                    combined_data["sections"][section_idx]["questions"].append(question)
                    question_map[question_key] = (section_idx, len(combined_data["sections"][section_idx]["questions"]) - 1)
        
        # First, sort by page number, then by question ID within each section
        for section in combined_data["sections"]:
            try:
                # Primary sort by page number, then by question ID within each page
                section["questions"].sort(
                    key=lambda q: (
                        q.get("_page_num", 999999),  # First by page number
                        int(''.join(filter(str.isdigit, q.get("id", "999999")))) if q.get("id") else 999999  # Then by numeric question ID
                    )
                )
            except (ValueError, TypeError):
                try:
                    # Fallback sort if numeric conversion fails
                    section["questions"].sort(
                        key=lambda q: (
                            q.get("_page_num", 999999),  # First by page number
                            q.get("id", "999999")  # Then by string question ID
                        )
                    )
                except:
                    # Last resort: sort just by page number
                    try:
                        section["questions"].sort(key=lambda q: q.get("_page_num", 999999))
                    except:
                        pass  # Give up if all sorting attempts fail
        
        return combined_data
    
    def convert_to_mutation_format(self, data: Dict[str, Any], provider: FeasibilityProvider) -> Dict[str, Any]:
        """
        Convert the internal data format to the mutation format
        
        Args:
            data: Structured survey data
            provider: Identified provider
            
        Returns:
            Data in mutation format
        """
        questions_with_index = []
        
        # Keep track of seen questions to avoid duplication in the final output
        seen_questions = set()
        
        for section_idx, section in enumerate(data.get("sections", [])):
            for question_idx, question in enumerate(section.get("questions", [])):
                # Extract page number for sorting
                page_num = question.get("_page_num", 999999)
                
                # Try to extract numeric ID for secondary sorting
                if "id" in question:
                    try:
                        id_num = int(''.join(filter(str.isdigit, question.get("id", "999999"))))
                        sort_key = id_num
                    except (ValueError, TypeError):
                        sort_key = 999999
                else:
                    sort_key = 999999
                
                # Get question text
                question_text = question.get("text", "").strip()
                
                # For fields that commonly appear multiple times, ensure we treat them as distinct
                # by creating a more specific identifier
                question_context = ""
                if question_text.lower() in ["email:", "email", "first name:", "first name", 
                                             "last name:", "last name"]:
                    question_context = f"{section.get('title', '')}:{section_idx}:{question.get('id', '')}"
                    
                # Create a unique identifier for this question
                question_unique_id = f"{question_context}:{question_text}"
                    
                # Skip if we've already processed this exact question
                if question_unique_id in seen_questions:
                    continue
                    
                seen_questions.add(question_unique_id)
                    
                questions_with_index.append({
                    "question": question,
                    "page_num": page_num,  # Store page number for primary sorting
                    "sort_key": sort_key,  # Secondary sort key
                    "section_idx": section_idx,
                    "question_idx": question_idx
                })
        
        # Sort first by page number, then by question ID
        questions_with_index.sort(key=lambda x: (x["page_num"], x["sort_key"]))
        
        entries = []
        for item in questions_with_index:
            question = item["question"]
            question_text = question.get("text", "").strip()
            
            # Ensure we have a consistent format for selected answers
            selected = question.get("selected")
            if selected is None:
                answers = []
            elif isinstance(selected, list):
                # Filter out empty or None values and keep only unique answers
                answers = [ans for ans in selected if ans]
            elif selected:  # Handle single string answers
                answers = [selected]
            else:
                answers = []
            
            # Remove duplicates while preserving order
            unique_answers = []
            for ans in answers:
                if ans not in unique_answers:
                    unique_answers.append(ans)
            
            entries.append({
                "question": question_text,
                "answers": unique_answers
            })
        
        mutation_data = {
            "entries": entries,
            "provider": provider.value,
            "id": str(uuid.uuid4())
        }
        
        return mutation_data 