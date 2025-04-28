"""
PDF utility functions for the survey parser application.
"""

import os
from asyncio import to_thread
from PyPDF2 import PdfReader, PdfWriter

async def get_pdf_page_count_async(pdf_path):
    """Get the number of pages in a PDF asynchronously"""
    def _get_page_count(path):
        reader = PdfReader(path)
        return len(reader.pages)
    
    # Run the blocking operation in a separate thread
    return await to_thread(_get_page_count, pdf_path)

async def extract_single_page_pdf_async(pdf_path, page_num, temp_dir):
    """
    Extract a single page from a PDF file and save it as a new PDF asynchronously
    """
    def _extract_page(path, p_num, t_dir):
        reader = PdfReader(path)
        writer = PdfWriter()
        
        # PyPDF2 uses 0-based indexing for pages
        if p_num <= len(reader.pages):
            writer.add_page(reader.pages[p_num - 1])
            
            # Save the extracted page as a new PDF
            output_path = os.path.join(t_dir, f"page_{p_num}.pdf")
            with open(output_path, "wb") as f:
                writer.write(f)
            
            return output_path
        else:
            print(f"Error: Page {p_num} does not exist in the PDF.")
            return None
    
    # Run the blocking operation in a separate thread
    return await to_thread(_extract_page, pdf_path, page_num, temp_dir)

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 500) -> list:
    """
    Split text into overlapping chunks of specific size
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of tuples containing (chunk_text, position)
    """
    if not text:
        return []
        
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [(text, 0)]
    
    chunks = []
    position = 0
    
    while position < len(text):
        if position + chunk_size >= len(text):
            chunk = text[position:]
            chunks.append((chunk, position))
            break
        
        end = position + chunk_size
        if end < len(text):
            newline_pos = text.rfind('\n', position + chunk_size - overlap, end)
            if newline_pos > position:
                end = newline_pos + 1
            else:
                space_pos = text.rfind(' ', position + chunk_size - overlap, end)
                if space_pos > position:
                    end = space_pos + 1
        
        chunk = text[position:end]
        chunks.append((chunk, position))
        
        position = end - overlap if end - overlap > position else end
    
    return chunks 