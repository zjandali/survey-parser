"""
Client for the LLMWhisperer API for OCR processing.
"""

import time
import asyncio
import aiohttp
from typing import Optional, Dict, Any

from config.settings import (
    LLMWHISPERER_API_KEY,
    WHISPER_API_URL,
    WHISPER_STATUS_URL,
    WHISPER_RETRIEVE_URL
)

class LLMWhispererClient:
    """Client for the LLMWhisperer API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or LLMWHISPERER_API_KEY
        if not self.api_key:
            raise ValueError("LLMWhisperer API key is required")
    
    async def submit_document(self, file_data: bytes, params: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Submit a document for OCR processing
        
        Args:
            file_data: Binary file data
            params: API parameters
            session: aiohttp session
            
        Returns:
            Response JSON with whisper_hash
        """
        headers = {
            "unstract-key": self.api_key,
            "Content-Type": "application/octet-stream"
        }
        
        async with session.post(
            WHISPER_API_URL,
            headers=headers,
            params=params,
            data=file_data
        ) as response:
            if response.status != 202:
                response_text = await response.text()
                raise Exception(f"Error submitting document: {response.status} - {response_text}")
            
            return await response.json()
    
    async def check_status(self, whisper_hash: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Check processing status
        
        Args:
            whisper_hash: Hash from submission
            session: aiohttp session
            
        Returns:
            Status response
        """
        headers = {"unstract-key": self.api_key}
        
        # Try direct URL format first
        async with session.get(
            f"{WHISPER_STATUS_URL}/{whisper_hash}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
        
        # Try alternative format with query parameter
        async with session.get(
            WHISPER_STATUS_URL,
            headers=headers,
            params={"whisper_hash": whisper_hash}
        ) as response:
            if response.status != 200:
                raise Exception(f"Error checking status: {response.status}")
            
            return await response.json()
    
    async def retrieve_results(self, whisper_hash: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Retrieve processing results
        
        Args:
            whisper_hash: Hash from submission
            session: aiohttp session
            
        Returns:
            Results data
        """
        headers = {"unstract-key": self.api_key}
        
        # Try direct URL format first
        async with session.get(
            f"{WHISPER_RETRIEVE_URL}/{whisper_hash}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
        
        # Try alternative format with query parameter
        async with session.get(
            WHISPER_RETRIEVE_URL,
            headers=headers,
            params={"whisper_hash": whisper_hash}
        ) as response:
            if response.status != 200:
                raise Exception(f"Error retrieving results: {response.status}")
            
            return await response.json()
    
    async def extract_text_with_polling(
        self, 
        file_data: bytes,
        session: aiohttp.ClientSession,
        wait_timeout: int = 300,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Submit document and poll until processing completes
        
        Args:
            file_data: Binary file data
            session: aiohttp session
            wait_timeout: Maximum wait time in seconds
            params: Optional API parameters
            
        Returns:
            Extracted text or None if processing failed
        """
        # Default parameters for OCR
        if params is None:
            params = {
                "mode": "form",
                "output_mode": "layout_preserving",
                "add_line_nos": 1
            }
        
        try:
            # Submit document
            result = await self.submit_document(file_data, params, session)
            whisper_hash = result["whisper_hash"]
            
            # Poll for completion with adaptive polling
            start_time = time.time()
            poll_interval = 0.5  # Start with a short interval
            
            while time.time() - start_time < wait_timeout:
                try:
                    status = await self.check_status(whisper_hash, session)
                    current_status = status.get('status', 'unknown')
                    
                    if current_status == "processed" or current_status == "delivered":
                        # Retrieve results
                        final_result = await self.retrieve_results(whisper_hash, session)
                        
                        # Extract text from response structure
                        if "extraction" in final_result:
                            return final_result["extraction"]["result_text"]
                        elif "result_text" in final_result:
                            return final_result["result_text"]
                        elif "text" in final_result:
                            return final_result["text"]
                        else:
                            raise Exception("Could not find result text in response")
                            
                    elif current_status == "processing":
                        # Adaptive polling when processing
                        poll_interval = min(poll_interval * 1.2, 2)
                        await asyncio.sleep(poll_interval)
                    elif current_status == "unknown":
                        raise Exception(f"Unknown status for whisper job")
                    else:
                        # Use exponential backoff for unknown statuses
                        poll_interval = min(poll_interval * 1.5, 3)
                        await asyncio.sleep(poll_interval)
                        
                except Exception as e:
                    print(f"Error polling: {str(e)}")
                    # Use exponential backoff for errors
                    poll_interval = min(poll_interval * 2, 5)
                    await asyncio.sleep(poll_interval)
            
            # If timeout is reached
            raise Exception(f"Processing timed out after {wait_timeout} seconds")
                
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None 