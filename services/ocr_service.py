"""
Service for extracting text from PDF pages using OCR.
"""

import os
import aiohttp
from pathlib import Path
from typing import Optional, Tuple

from core.file_operations import read_file_async, remove_file_async
from core.pdf_utils import extract_single_page_pdf_async
from api.llm_whisperer import LLMWhispererClient

class OCRService:
    """Service for OCR text extraction"""
    
    def __init__(self):
        """Initialize the OCR service"""
        self.llm_whisperer = LLMWhispererClient()
    
    async def extract_text_from_page(
        self,
        file_path: str, 
        page_num: int, 
        temp_dir: str, 
        session: aiohttp.ClientSession, 
        wait_timeout: int = 300
    ) -> Optional[str]:
        """
        Extract text from a specific page of a document using OCR
        
        Args:
            file_path: Path to the file (PDF, DOC, DOCX)
            page_num: Page number to extract (1-based) - only used for PDFs
            temp_dir: Directory for temporary files
            session: aiohttp session for API requests
            wait_timeout: Maximum wait time in seconds
            
        Returns:
            Extracted text or None if extraction failed
        """
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()
        
        print(f"Extracting text from {'page ' + str(page_num) if file_extension == '.pdf' else 'document'}... using LLMwhisperer")
        
        try:
            # Handle different file types
            if file_extension == '.pdf':
                # Extract the single page as a separate PDF
                single_page_pdf = await extract_single_page_pdf_async(file_path, page_num, temp_dir)
                
                if not single_page_pdf:
                    return None
                
                # Read the PDF file into binary data
                file_data = await read_file_async(single_page_pdf, mode='rb')
                
                # Clean up after use
                cleanup_path = single_page_pdf
            else:
                # For non-PDF files, process the entire document directly
                file_data = await read_file_async(file_path, mode='rb')
                cleanup_path = None
            
            # Submit to LLMWhisperer for OCR processing
            ocr_text = await self.llm_whisperer.extract_text_with_polling(
                file_data=file_data,
                session=session,
                wait_timeout=wait_timeout
            )
            
            # Clean up temporary file if needed
            if cleanup_path:
                await remove_file_async(cleanup_path)
            
            return ocr_text
            
        except Exception as e:
            print(f"Error extracting text from {'page ' + str(page_num) if file_extension == '.pdf' else 'document'}: {str(e)}")
            return None 