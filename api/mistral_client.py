"""
Client for the Mistral API for LLM processing.
"""

import asyncio
from typing import Dict, Any, List, Optional
from mistralai import Mistral
from config.settings import MISTRAL_API_KEY

class MistralClient:
    """Client for the Mistral API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or MISTRAL_API_KEY
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        self.client = Mistral(api_key=self.api_key)
    
    async def complete_async(
        self, 
        prompt: str, 
        model: str = "mistral-large-latest",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ) -> str:
        """
        Generate text completion asynchronously
        
        Args:
            prompt: The prompt to complete
            model: The model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.chat.complete_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
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

    CRITICALLY IMPORTANT: 
    - Do NOT alter, reformat, clean, or pre-process the OCR text in any way before analyzing it
    - The exact spatial layout and positioning of text elements are essential for correctly identifying questions and answers
    - Checkbox positions, indentation, line breaks, and other formatting details are crucial for correctly interpreting the form structure
    - Preserve all spacing, line breaks, and text positioning exactly as provided in the OCR text

    CRITICAL SPATIAL AWARENESS INSTRUCTIONS:
    - Survey forms often have complex layouts where answers don't always appear directly below or beside their questions
    - Scan the ENTIRE document space including headers, footers, margins, and floating elements for potential answers
    - Pay special attention to:
      * Headers/letterheads (often contain institution/company names, dates, ID numbers)
      * Floating text boxes/fields (may contain answers to questions that appear elsewhere)
      * Text in different columns that may be related
      * Information in margins or page corners
      * Titles and subtitles that contain relevant information
    - Be aware that some answers may appear visually separated from their questions in the original form
    - For questions asking about names, addresses, dates, or identifiers, check page headers and document metadata
    - Match answers to questions based on context and content, not just proximity

    Very important rules:
    - Ensure you extract EVERY question on the page, even if it appears to be a continuation from another page
    - Pay special attention to formatting like checkboxes, bullet points, and asterisks
    - If there are multiple checked boxes for a single question, make sure to include all selected options as an array
    - For free-text responses, include the complete answer text
    - If a question appears incomplete (continuing to another page), add "[INCOMPLETE]" to the text field

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
      * Headers/letterheads (often contain institution/company names, dates, ID numbers)
      * Floating text boxes/fields (may contain answers to questions that appear elsewhere)
      * Text in different columns that may be related
      * Information in margins or page corners
      * Titles and subtitles that contain relevant information
    - Be aware that some answers may appear visually separated from their questions in the original form
    - For questions asking about names, addresses, dates, or identifiers, check page headers and document metadata
    - Match answers to questions based on context and content, not just proximity
    - When dealing with page breaks ('<<<'), remember that important contextual information might be split across pages

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
        prompt = f"""Identify the survey provider from the OCR text. You MUST return ONLY ONE of these exact values (no additional text):
SURVEY_MONKEY
GOOGLE_FORMS
QUALTRICS
PDF
MICROSOFT_WORD
MICROSOFT_FORMS
NULL

Look for indicators such as:
- Provider-specific layouts, logos or branding elements
- Platform-specific question formats or styling
- URLs or email domains related to the provider
- Copyright notices or footers mentioning the provider
- Form structure that matches a specific platform

IMPORTANT DISTINCTIONS:
- Microsoft Word documents often have checkboxes ([ ] or [X]) for forms - this does NOT mean it's Microsoft Forms
- Microsoft Forms will specifically have a forms.office.com URL or Microsoft Forms branding
- Look for file paths with .doc or .docx extensions which indicate Microsoft Word documents
- Microsoft Word forms typically have more complex/custom layouts and formatting than Microsoft Forms
- Microsoft Forms has a very specific look with consistent question styling and automatic numbering
- Only classify as MICROSOFT_FORMS if you see clear evidence of the Microsoft Forms platform

If you can't identify a specific provider, respond with "PDF" for PDF files or "MICROSOFT_WORD" for Word documents, based on the file type.
If you can't determine anything, respond with "NULL".

This is a {file_extension.upper()} document{" - " + chunk_info if is_chunk else ""} with the following OCR text:
{text[:3000]}

IMPORTANT: Return ONLY ONE of the allowed values with no additional text or explanation.
Survey Provider:
"""
        return prompt 