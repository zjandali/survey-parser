"""
Client for DeepSeek R1 model via OpenRouter API using OpenAI client.
"""

import asyncio
import logging
import random
import time
import re
import json
import base64
from typing import Dict, Any, List, Optional, Union
import os
from openai import AsyncOpenAI
from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_RETRY_BASE_DELAY,
    OPENROUTER_MAX_RETRIES,
    SITE_URL,
    SITE_NAME
)

class DeepSeekClient:
    """Client for DeepSeek R1 model via OpenRouter API using OpenAI client"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required for DeepSeek R1 access")
        
        # Initialize the OpenAI client with OpenRouter base URL
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # Headers for OpenRouter
        self.headers = {
            "HTTP-Referer": SITE_URL or "https://survey-parser.example.com",
            "X-Title": SITE_NAME or "Survey Parser"
        }
        
        # Track request timestamps for adaptive rate limiting
        self.recent_requests = []
        self.rate_window = 60  # Track requests in a 60-second window
        self.rate_limit = 100  # Conservative limit for OpenRouter
    
    async def _adaptive_rate_limit(self):
        """
        Adaptively manage rate limiting based on recent request patterns
        """
        # Clean up old requests outside the window
        current_time = time.time()
        self.recent_requests = [t for t in self.recent_requests if current_time - t <= self.rate_window]
        
        # If approaching rate limit (>70% of limit), add adaptive delay
        if len(self.recent_requests) > 0.7 * self.rate_limit:
            # Calculate a delay that adapts to how close we are to the limit
            usage_ratio = len(self.recent_requests) / self.rate_limit
            adaptive_delay = max(0.2, usage_ratio * 3 - 1.5)
            logging.info(f"Rate limit management: {len(self.recent_requests)}/{self.rate_limit} requests in window. Waiting {adaptive_delay:.2f}s")
            await asyncio.sleep(adaptive_delay)
    
    def _clean_json_content(self, text: str) -> str:
        """
        Clean and format JSON content from DeepSeek R1 model output
        
        Args:
            text: Raw text from the model
            
        Returns:
            Cleaned text with properly formatted JSON
        """
        # First strip any markdown code block indicators
        cleaned = re.sub(r'```json\s+', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Check for JSON content between braces
        json_match = re.search(r'(\{[\s\S]*\})', cleaned)
        if json_match:
            potential_json = json_match.group(1)
            # Try to validate it's proper JSON
            try:
                # Parse and re-serialize to ensure proper formatting
                parsed = json.loads(potential_json)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                # If we can't parse, try to fix common issues
                fixed_json = potential_json
                
                # Replace single quotes with double quotes
                fixed_json = fixed_json.replace("'", '"')
                
                # Fix unquoted keys (property names should be in double quotes)
                fixed_json = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', fixed_json)
                
                # Fix trailing commas in arrays and objects
                fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)
                
                try:
                    # Try parsing again
                    parsed = json.loads(fixed_json)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    # Still not valid JSON, return the original text
                    logging.warning("Unable to fix JSON format in DeepSeek R1 response")
                    return text
        return text
    
    async def complete_async(
        self, 
        prompt: str, 
        model: str = "deepseek/deepseek-r1",
        max_tokens: int = 1000000,  # Changed from 4000 to 1000000 (effectively no limit)
        temperature: float = 0.1,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> str:
        """
        Generate text completion asynchronously with rate limiting and exponential backoff
        
        Args:
            prompt: The prompt to complete
            model: The model to use (default: deepseek/deepseek-r1)
            max_tokens: Maximum tokens to generate (default: 1000000, effectively no limit)
            temperature: Temperature for generation
            image_path: Optional path to local image file to include with the prompt
            image_url: Optional URL of an image to include with the prompt
            
        Returns:
            Generated text
        """
        # Apply adaptive rate limiting
        await self._adaptive_rate_limit()
        
        attempt = 0
        base_delay = OPENROUTER_RETRY_BASE_DELAY
        
        while True:  # Infinite retry loop
            try:
                # Track this request
                self.recent_requests.append(time.time())
                
                # Prepare messages
                if image_path or image_url:
                    # For image inputs, we use the content list format
                    content = [{"type": "text", "text": prompt}]
                    
                    # Add image from URL if provided
                    if image_url:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    
                    # Add image from local file if provided
                    elif image_path:
                        # Convert image to base64
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    
                    # Create message with content list
                    messages = [{"role": "user", "content": content}]
                else:
                    # For text-only inputs, use standard message format
                    messages = [{"role": "user", "content": prompt}]
                
                # Make the API call with OpenAI client to OpenRouter
                response = await self.client.chat.completions.create(
                    extra_headers=self.headers,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Extract text response
                raw_content = response.choices[0].message.content
                
                # Add debug logging for provider classification
                # Check for both old and new prompt formats
                if "Survey Provider:" in prompt or "ALLOWED RESPONSES - Choose exactly ONE of these:" in prompt:
                    logging.info(f"DeepSeek raw provider response: '{raw_content}'")
                    if not raw_content.strip():
                        logging.warning("Empty response received from LLM")
                        # Don't automatically return a fallback value - let the caller handle this
                        # This keeps behavior consistent with other providers
                
                # Check if this looks like it should be JSON
                if '```json' in prompt or '"sections"' in prompt or prompt.strip().endswith("json"):
                    # This is likely expected to return JSON, so clean and format it
                    return self._clean_json_content(raw_content)
                else:
                    # Regular text response
                    return raw_content
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error or server overload
                is_rate_limit = "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str
                is_server_overload = "503" in error_str or "overloaded" in error_str or "not ready" in error_str
                
                if is_rate_limit or is_server_overload:
                    # Log the error
                    if is_rate_limit:
                        logging.warning(f"DeepSeek R1 rate limit error: '{str(e)[:100]}...")
                    else:
                        logging.warning(f"DeepSeek R1 server overload error: '{str(e)[:100]}...")
                    
                    # Calculate exponential backoff time with jitter
                    delay = base_delay * (2 ** min(attempt, 10)) + random.uniform(0, 1.0)
                    
                    # Add extra delay for server overload
                    if is_server_overload:
                        delay = delay * 1.5  # 50% longer delay for server issues
                        
                    logging.warning(f"DeepSeek R1 API error. Retrying in {delay:.2f} seconds (attempt {attempt+1}/∞)")
                    await asyncio.sleep(delay)
                    attempt += 1
                else:
                    # For other errors, also retry (with a warning)
                    logging.warning(f"Error calling DeepSeek R1 API: {str(e)}. Will retry...")
                    await asyncio.sleep(base_delay * (2 ** min(attempt, 5)))
                    attempt += 1
    
    async def process_image_with_text(
        self,
        image_path: str,
        prompt: str,
        model: str = "deepseek/deepseek-r1",
        max_tokens: int = 1000000,
        temperature: float = 0.1
    ) -> str:
        """
        Process an image with accompanying text prompt
        
        Args:
            image_path: Path to the image file
            prompt: The text prompt to accompany the image
            model: The model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text
        """
        return await self.complete_async(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            image_path=image_path
        )
    
    def get_extraction_prompt(self, ocr_text: str, page_info: str = "") -> str:
        """
        Create a prompt for extracting structured data
        
        Args:
            ocr_text: The OCR text to analyze
            page_info: Additional page context
            
        Returns:
            Formatted extraction prompt
        """
        context = ""
        if page_info:
            context = f"""
This is a single page ({page_info}) from a larger document.
Extract all questions and answers found on this specific page only.
Pay special attention to questions at the end of the page that might continue to the next page or answers that appear at the top of this page that belong to questions from the previous page.
"""

        return f"""
You are an expert at analyzing survey data from OCR-processed documents. 

Your task is to extract all questions and answers from the following survey page in a structured format.
{context}

For each question:
1. Extract the question ID (e.g., Q2.3, Q3.1) if present. If no ID is explicitly shown, generate a sequential one.
2. Extract the complete question text, including any asterisks (*) that denote required fields.
3. Identify all possible answer options.
4. Determine which answer was selected based on:
   - Checkboxes: Look for [X], ☑, or similar marks
   - Radio buttons: Look for filled circles or selected options
   - Text fields: Extract the text entered
   - Multiple selections: If multiple options are checked, include ALL of them as a list
5. Identify any section headers or organizational elements.

CRITICAL SPATIAL AWARENESS INSTRUCTIONS:
- Survey forms often have complex layouts where answers don't always appear directly below or beside their questions
- Scan the ENTIRE document space including headers, footers, margins, and floating elements for potential answers
- Pay special attention to:
  * Multi-column layouts where questions and answers may appear in separate columns
  * Headers/letterheads (often contain institution/company names, dates, ID numbers)
  * Floating text boxes/fields (may contain answers to questions that appear elsewhere)
  * Text in different columns that may be related
  * Information in margins or page corners
  * Titles and subtitles that contain relevant information
- Be aware that some answers may appear visually separated from their questions in the original form
- For questions asking about names, addresses, dates, or identifiers, check page headers and document metadata
- Match answers to questions based on context and content, not just proximity
- Watch for indented text blocks below questions (like "20 per month") which are likely the answers to those questions

MULTI-COLUMN LAYOUT INSTRUCTIONS:
- Many forms use side-by-side columns that get linearized in OCR text with large spaces
- When you see lots of spaces in a line, it likely indicates content from different columns
- Content from the left column often continues for several lines before content from the right column appears
- Text that appears indented under a question ID (like "11.") and has no other question nearby is likely the answer
- Look for answers that appear in the same column as their questions but may be separated by text from other columns
- IMPORTANT: When you see checkboxes ([X] or []) appearing at the end of lines after lots of spaces, they often belong to a different question in another column
- Indent patterns like "20 per month" appearing below a question are almost always the answers to that question, even if there's no checkbox
- For questions like "How many subjects can be enrolled per month?" look for numerical answers followed by time units (e.g., "20 per month")

Very important rules:
- Ensure you extract EVERY question on the page, even if it appears to be a continuation from another page
- Pay special attention to formatting like checkboxes, bullet points, and asterisks
- If there are multiple checked boxes for a single question, make sure to include all selected options as an array
- For free-text responses, include the complete answer text
- If a question appears incomplete (continuing to another page), add "[INCOMPLETE]" to the text field
- Look for special markers like [LIKELY_ANSWER_TO_Q19: 2+] which indicate answers to questions across pages
- If you see a standalone answer like "2+" or "Yes" at the top of the page, it might be an answer to a question from the previous page

YOU MUST FORMAT YOUR RESPONSE AS VALID JSON without any extra explanation. Return ONLY the JSON data.
Format your response in JSON structure as follows:
```json
{{
  "sections": [
    {{
      "title": "Section Title",
      "questions": [
        {{
          "id": "Q1.1",
          "text": "Full question text",
          "options": ["Option 1", "Option 2", "Option 3"],
          "selected": "Option 2"
        }},
        {{
          "id": "Q1.2",
          "text": "Multiple-select question text",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "selected": ["Option A", "Option C"]
        }},
        ...
      ]
    }},
    ...
  ]
}}
```

If you only see partial questions (continuing from a previous page or continuing to the next page), extract what you can and indicate with "[INCOMPLETE]" in the text field.

Accurately preserve all information while fixing any OCR artifacts or formatting issues. If something is ambiguous, indicate this with "[UNCLEAR]".

If you see a "Next Page Preview" section, use this information to understand the context of questions but don't extract questions from that preview section.

Here is the OCR text to analyze:

{ocr_text}
"""

    def get_chunked_extraction_prompt(self, ocr_text: str, chunk_info: str = "") -> str:
        """
        Create a prompt for extracting structured data from a text chunk that may contain
        content spanning multiple pages with clear page break markers (<<<)
        
        Args:
            ocr_text: The OCR text chunk to analyze
            chunk_info: Additional chunk context
            
        Returns:
            Formatted extraction prompt for chunk processing
        """
        context = ""
        if chunk_info:
            context = f"""
This is a chunk of text ({chunk_info}) that may span across page boundaries.
The chunk contains content from multiple pages with '<<<' markers indicating page breaks.
"""

        return f"""
You are an expert at analyzing survey data from OCR-processed documents.

Your task is to extract all questions and answers from the following survey text chunk in a structured format.
{context}

IMPORTANT CHUNK PROCESSING INSTRUCTIONS:
- This chunk may contain content from multiple pages
- The '<<<' markers indicate page breaks
- Pay special attention to questions and answers that span across these page break markers
- When you see a question right before a '<<<' marker, look for its answer right after the marker
- When you see an answer at the beginning of the chunk (before any question), it likely belongs to a question from the previous page

CRITICAL SPATIAL AWARENESS INSTRUCTIONS:
- Survey forms often have complex layouts where answers don't always appear directly below or beside their questions
- Scan the ENTIRE document space including headers, footers, margins, and floating elements for potential answers
- Pay special attention to:
  * Multi-column layouts where questions and answers may appear in separate columns
  * Headers/letterheads (often contain institution/company names, dates, ID numbers)
  * Floating text boxes/fields (may contain answers to questions that appear elsewhere)
  * Text in different columns that may be related
  * Information in margins or page corners
  * Titles and subtitles that contain relevant information
- Be aware that some answers may appear visually separated from their questions in the original form
- For questions asking about names, addresses, dates, or identifiers, check page headers and document metadata
- Match answers to questions based on context and content, not just proximity
- When dealing with page breaks ('<<<'), remember that important contextual information might be split across pages
- Watch for indented text blocks below questions (like "20 per month") which are likely the answers to those questions

MULTI-COLUMN LAYOUT INSTRUCTIONS:
- Many forms use side-by-side columns that get linearized in OCR text with large spaces
- When you see lots of spaces in a line, it likely indicates content from different columns
- Content from the left column often continues for several lines before content from the right column appears
- Text that appears indented under a question ID (like "11.") and has no other question nearby is likely the answer
- Look for answers that appear in the same column as their questions but may be separated by text from other columns
- IMPORTANT: When you see checkboxes ([X] or []) appearing at the end of lines after lots of spaces, they often belong to a different question in another column
- Indent patterns like "20 per month" appearing below a question are almost always the answers to that question, even if there's no checkbox
- For questions like "How many subjects can be enrolled per month?" look for numerical answers followed by time units (e.g., "20 per month")

For each question:
1. Extract the question ID (e.g., Q2.3, Q3.1) if present. If no ID is explicitly shown, generate a sequential one.
2. Extract the complete question text, including any asterisks (*) that denote required fields.
3. Identify all possible answer options.
4. Determine which answer was selected based on:
   - Checkboxes: Look for [X], ☑, or similar marks
   - Radio buttons: Look for filled circles or selected options
   - Text fields: Extract the text entered
   - Multiple selections: If multiple options are checked, include ALL of them as a list
5. Identify any section headers or organizational elements.

Very important rules:
- Pay special attention to cases where a question appears right before a '<<<' marker and its answer appears right after it
- Look for standalone answers at the top of a chunk that might belong to questions from the previous page
- For isolated answers like "2+" or "Yes" that appear at the start of the chunk, try to match them to the last question that appears in the previous section (before the first '<<<' marker)
- For questions at the end of the chunk (after the last '<<<' marker) that don't have answers, mark them as "[INCOMPLETE]"
- If there are multiple checked boxes for a question, include all selected options as an array
- For free-text responses, include the complete answer text

YOU MUST FORMAT YOUR RESPONSE AS VALID JSON without any extra explanation. Return ONLY the JSON data.
Format your response in JSON structure as follows:
```json
{{
  "sections": [
    {{
      "title": "Section Title",
      "questions": [
        {{
          "id": "Q1.1",
          "text": "Full question text",
          "options": ["Option 1", "Option 2", "Option 3"],
          "selected": "Option 2"
        }},
        {{
          "id": "Q1.2",
          "text": "Multiple-select question text",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "selected": ["Option A", "Option C"]
        }},
        ...
      ]
    }},
    ...
  ]
}}
```

If a question appears incomplete, extract what you can and indicate with "[INCOMPLETE]" in the text field.
If something is ambiguous, indicate this with "[UNCLEAR]".

Here is the OCR text chunk to analyze:

{ocr_text}
"""
    
    def get_classifier_prompt(self, text: str, file_extension: str, is_chunk: bool = False, chunk_info: str = "") -> str:
        """
        Generate a classification prompt for provider identification
        
        Args:
            text: The OCR text to analyze
            file_extension: File extension (e.g., "pdf")
            is_chunk: Whether this is a chunk of a larger document
            chunk_info: Information about the chunk position
            
        Returns:
            Complete prompt for provider classification
        """
        # Modified prompt specifically designed to match Gemini's format
        prompt = f"""You're evaluating a survey document to identify which platform or software created it.

TASK: Analyze the OCR text from this document and identify the survey provider.

ALLOWED RESPONSES - Choose exactly ONE of these:
- SURVEY_MONKEY
- GOOGLE_FORMS
- QUALTRICS
- PDF
- MICROSOFT_WORD
- MICROSOFT_FORMS
- NULL

KEY INDICATORS TO LOOK FOR:
- URLs containing domains like "surveymonkey.com", "forms.office.com", "forms.microsoft.com", "docs.google.com/forms", or "qualtrics.com"
- Copyright notices or footers mentioning any of these providers
- Provider-specific layouts, question formatting, or styling patterns
- Branding elements or logos from these platforms
- Form structure characteristics unique to specific platforms

IMPORTANT DISTINCTIONS:
- Microsoft Word documents often have checkboxes ([ ] or [X]) for forms - this does NOT mean it's Microsoft Forms
- Microsoft Forms will specifically have a forms.office.com URL or Microsoft Forms branding
- Look for file paths with .doc or .docx extensions which indicate Microsoft Word documents
- Microsoft Word forms typically have more complex/custom layouts and formatting than Microsoft Forms
- Microsoft Forms has a very specific look with consistent question styling and automatic numbering

DOCUMENT DETAILS:
This is a {file_extension.upper()} document{" - " + chunk_info if is_chunk else ""}.

DOCUMENT OCR TEXT:
{text[:3000]}

INSTRUCTIONS:
1. Respond with ONLY ONE of the allowed values listed above
2. Do not include any explanations, additional text, or punctuation
3. Only classify as "MICROSOFT_FORMS" if you see clear evidence of Microsoft Forms (forms.office.com URLs or Microsoft Forms branding)
4. The presence of checkboxes or form elements alone is NOT sufficient to classify as Microsoft Forms
5. If you find file paths containing .doc or .docx, this strongly indicates MICROSOFT_WORD
6. If you find no specific platform indicators, respond with "PDF" for PDF files or "MICROSOFT_WORD" for Word documents
7. If you can't determine anything at all, respond with "NULL"

Your response:"""
        return prompt 