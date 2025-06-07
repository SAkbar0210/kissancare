import json
import os
import re

# Assuming translations.json is in the same directory as utils.py
TRANSLATIONS_FILE = 'translations.json'
TRANSLATIONS = {}

def load_translations():
    """Loads translations from the JSON file."""
    global TRANSLATIONS
    if not TRANSLATIONS: # Load only if not already loaded
        try:
            with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
                TRANSLATIONS = json.load(f)
        except FileNotFoundError:
            print(f"Error: Translations file not found at {TRANSLATIONS_FILE}")
            TRANSLATIONS = {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {TRANSLATIONS_FILE}")
            TRANSLATIONS = {}

def get_translation(term, target_language):
    """
    Translates a term to the target language using loaded translations.
    Falls back to the original term if translation is not found.
    """
    load_translations() # Ensure translations are loaded
    lang_translations = TRANSLATIONS.get(target_language, {})
    return lang_translations.get(term, term) # Default to term if not found


def is_valid_indian_phone_number(phone):
    """
    Validates if a string is a valid 10-digit Indian mobile number.
    Optionally handles the +91 prefix.
    """
    # Remove +91 prefix if it exists
    if phone.startswith('+91'):
        phone = phone[3:]
        # Remove any space after +91 if present
        if phone.startswith(' '):
            phone = phone[1:]

    # Define the regex pattern for a 10-digit number starting with 6, 7, 8, or 9
    pattern = re.compile(r'^[6789]\d{9}$')

    # Check if the cleaned phone number matches the pattern
    if pattern.match(phone):
        return True
    else:
        return False

# You might also define constants like DISEASE_NUTRIENT_INTERACTIONS here
# DISEASE_NUTRIENT_INTERACTIONS = {...} 