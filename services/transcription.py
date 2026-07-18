import re

def correct_grammar(text: str) -> str:
    """Apply rule-based grammar and contraction corrections to raw transcribed text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Dictionary of common contraction typos/slang
    contraction_fixes = {
        r"\b(i|I)'?m\b": "I'm",
        r"\b(d|D)ont\b": "don't",
        r"\b(c|C)ant\b": "can't",
        r"\b(w|W)ont\b": "won't",
        r"\b(i|I)d\b": "I'd",
        r"\b(i|I)ve\b": "I've",
        r"\b(i|I)ll\b": "I'll",
        r"\b(d|D)idnt\b": "didn't",
        r"\b(w|W)asnt\b": "wasn't",
        r"\b(w|W)erent\b": "weren't",
        r"\b(i|I)snt\b": "isn't",
        r"\b(a|A)rent\b": "aren't",
        r"\b(h|H)asnt\b": "hasn't",
        r"\b(h|H)avent\b": "haven't",
        r"\b(c|C)ouldnt\b": "couldn't",
        r"\b(s|S)houldnt\b": "shouldn't",
        r"\b(w|W)ouldnt\b": "wouldn't",
        r"\b(t|T)hats\b": "that's",
        r"\b(i|I)ts\b": "its",  # Note: "its" vs "it's" is context-dependent, we default to "it's" for common speech if followed by adj/verb
        r"\b(i|I)t'?s\b": "it's",
    }
    
    for pattern, replacement in contraction_fixes.items():
        text = re.sub(pattern, replacement, text)
        
    # Capitalize standalone "i" pronoun (e.g. "what i want" -> "what I want")
    text = re.sub(r"\bi\b", "I", text)
    
    # Clean up punctuation spacing: e.g., "hello , world" -> "hello, world"
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    
    # Ensure there is exactly one space after punctuation if followed by a character
    text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)
    
    # Capitalize first letter of each sentence
    # Split by sentence boundaries, capitalize, and re-join
    sentences = re.split(r"([.!?]\s*)", text)
    capitalized_sentences = []
    capitalize_next = True
    
    for part in sentences:
        if not part:
            continue
        if re.match(r"^[.!?]\s*$", part):
            capitalized_sentences.append(part)
            capitalize_next = True
        else:
            part_str = part.strip()
            if part_str and capitalize_next:
                # Find first letter to capitalize
                for idx, char in enumerate(part):
                    if char.isalpha():
                        part = part[:idx] + char.upper() + part[idx+1:]
                        break
                capitalize_next = False
            capitalized_sentences.append(part)
            
    text = "".join(capitalized_sentences)
    
    # Capitalize days of the week and months
    proper_nouns = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", 
        "September", "October", "November", "December"
    ]
    for noun in proper_nouns:
        text = re.sub(rf"\b{noun.lower()}\b", noun, text)
        
    return text.strip()


def polish_transcript(text: str) -> str:
    """Apply safe punctuation/capitalization cleanup and grammar correction without changing review meaning."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    
    # Run our rule-based grammar correction first
    text = correct_grammar(text)
    
    # Ensure it starts capitalized and ends with a punctuation
    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
            
    return text

