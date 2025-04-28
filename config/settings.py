"""
Configuration settings for the survey parser application.
"""

import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)
else:
    print("Warning: .env file not found.")



LLMWHISPERER_API_KEY = os.getenv("LLMWHISPERER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ensure API keys are available
if not LLMWHISPERER_API_KEY:
    raise ValueError(
        "LLMWHISPERER_API_KEY environment variable not set or found in .env. "
        "Please set it with your LLMWhisperer API key."
    )

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY environment variable not set or found in .env. "
        "Please set it with your Mistral API key."
    )

if not TOGETHER_API_KEY:
    print("Warning: TOGETHER_API_KEY not set. Together AI workflow will not be available.")

if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY not set. OpenRouter workflow will not be available.")

if LLMWHISPERER_API_KEY:
    masked_key = f"{LLMWHISPERER_API_KEY[:4]}...{LLMWHISPERER_API_KEY[-4:]}" if len(LLMWHISPERER_API_KEY) > 8 else "[too short]"
    print(f"Using LLMWhisperer API key: {masked_key}")

WHISPER_API_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper"
WHISPER_STATUS_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper-status"
WHISPER_RETRIEVE_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper-retrieve"

OUTPUT_DIR = "output"

DEFAULT_TIMEOUT = 300
DEFAULT_FILE_PATTERN = "*.pdf,*.doc,*.docx"
DEFAULT_MAX_RAM_PERCENT = 80.0
DEFAULT_LLM_PROVIDER = "gemini"

# Rate limiting configuration
# Together rate limits for Tier 1: 600 RPM (10 RPS)
TOGETHER_CONCURRENCY_LIMIT = 5  # Increased to utilize Tier 1 limits better
TOGETHER_REQUEST_DELAY = 0.2  # Allow for ~5 requests per second (300 RPM), provides safety margin
TOGETHER_RETRY_BASE_DELAY = 1.0  # Base delay for exponential backoff
TOGETHER_MAX_RETRIES = 5  # Maximum number of retries for rate limited requests

# Mistral rate limits are higher
MISTRAL_CONCURRENCY_LIMIT = 5

# OpenRouter Settings
SITE_URL = os.getenv("SITE_URL", "https://survey-parser.example.com")
SITE_NAME = os.getenv("SITE_NAME", "Survey Parser")

# OpenRouter Concurrency limits
OPENROUTER_CONCURRENCY_LIMIT = 3  # Conservative limit for OpenRouter

# OpenRouter Retry settings
OPENROUTER_RETRY_BASE_DELAY = 1.0
OPENROUTER_MAX_RETRIES = 5

# DeepSeek R1 settings (via OpenRouter)
DEEPSEEK_CONCURRENCY_LIMIT = 3  # Conservative limit for DeepSeek R1

# Claude settings (via OpenRouter)
CLAUDE_CONCURRENCY_LIMIT = 2  # Conservative limit for Claude

# Gemini settings (via OpenRouter)
GEMINI_CONCURRENCY_LIMIT = 2  # Conservative limit for Gemini