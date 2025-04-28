"""
Asynchronous file operations for the survey parser application.
"""

import os
import shutil
from asyncio import to_thread

async def create_directory_async(directory):
    """Create a directory asynchronously"""
    def _create_dir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    return await to_thread(_create_dir, directory)

async def read_file_async(file_path, mode='r', encoding='utf-8'):
    def _read_file(path, m, enc):
        if 'b' in m:  # binary mode
            with open(path, mode=m) as f:
                return f.read()
        else:  # text mode
            with open(path, mode=m, encoding=enc) as f:
                return f.read()
    
    return await to_thread(_read_file, file_path, mode, encoding)

async def write_file_async(file_path, content, mode='w', encoding='utf-8'):
    def _write_file(path, data, m, enc):
        with open(path, mode=m, encoding=enc) as f:
            f.write(data)
    
    await to_thread(_write_file, file_path, content, mode, encoding)

async def remove_file_async(file_path):
    def _remove_file(path):
        try:
            os.remove(path)
            return True
        except Exception as e:
            print(f"Error removing file {path}: {str(e)}")
            return False
    
    return await to_thread(_remove_file, file_path)

async def rmtree_async(directory):
    def _rmtree(dir_path):
        try:
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            print(f"Error removing directory {dir_path}: {str(e)}")
            return False
    
    return await to_thread(_rmtree, directory)

async def create_results_dir_async():
    results_dir = "results"
    await create_directory_async(results_dir)
    
    intermediate_dir = os.path.join(results_dir, "ocr_text")
    await create_directory_async(intermediate_dir)
    
    temp_dir = os.path.join(results_dir, "temp")
    await create_directory_async(temp_dir)
    
    return results_dir, intermediate_dir, temp_dir 