"""
Text processing utilities.
"""

from typing import List, Tuple

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 500) -> List[Tuple[str, int]]:
    """
    Split text into overlapping chunks to ensure context is preserved
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Overlap size between chunks
        
    Returns:
        List of (chunk_text, start_position) tuples
    """
    # If the text is smaller than the chunk size, return it as is
    if len(text) <= chunk_size:
        return [(text, 0)]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Find end position with potential overlap
        end = min(start + chunk_size, len(text))
        
        # If this isn't the last chunk and we're not at the end of the text
        if end < len(text):
            # Try to find a natural break like newline, period or space
            for break_char in ['\n\n', '\n', '. ', ' ']:
                # Look for the last occurrence of break_char within a window at the end
                # to avoid cutting in the middle of a sentence or paragraph
                break_window = text[end - min(overlap, end - start):end]
                last_break = break_window.rfind(break_char)
                
                if last_break != -1:
                    # Adjust the end position to include the break character
                    end = end - (len(break_window) - last_break - len(break_char))
                    break
        
        # Add the chunk with its starting position
        chunks.append((text[start:end], start))
        
        # Move to the next position, accounting for overlap
        if end == len(text):
            break
            
        start = end - overlap if overlap < end else 0
        
        # Ensure we make progress and avoid infinite loops
        if start >= end:
            start = end
    
    return chunks 