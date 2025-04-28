"""
Enums for the survey parser application.
"""

from enum import Enum

class FeasibilityProvider(str, Enum):
    """
    Enum for different types of survey providers.
    Used for classifying the source of survey data.
    """
    SURVEY_MONKEY = "SURVEY_MONKEY"
    GOOGLE_FORMS = "GOOGLE_FORMS"
    QUALTRICS = "QUALTRICS"
    PDF = "PDF"
    MICROSOFT_WORD = "MICROSOFT_WORD"
    MICROSOFT_FORMS = "MICROSOFT_FORMS"
    NULL = "NULL" 