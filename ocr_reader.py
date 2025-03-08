from multiprocessing import Pool
import os
import cv2
import json
from datetime import datetime
from paddleocr import PaddleOCR
from utils import (
    create_annotated_image,
    FIELD_LABELS,
    COUNTRY_CODES,
    EXCLUDED_WORDS,
    SUPPORTED_LANGUAGES,
    logger,
    enhance_image
)

class HealthCardInfo:
    def __init__(self):
        self.insurance_number = ""
        self.surname = ""
        self.first_name = ""
        self.birth_date = ""
        self.personal_number = ""
        self.insurance_code = ""  # 4-digit Swiss insurance code
        self.insurance_name = ""  # Insurance provider name
        self.card_number = ""
        self.expiry_date = ""
        self.detected_language = ""

    def to_dict(self):
        return {
            "insurance_number": self.insurance_number,
            "surname": self.surname,
            "first_name": self.first_name,
            "birth_date": self.birth_date,
            "personal_number": self.personal_number,
            "insurance_code": self.insurance_code,
            "insurance_name": self.insurance_name,
            "card_number": self.card_number,
            "expiry_date": self.expiry_date,
            "detected_language": self.detected_language
        }

def detect_card_language(results):
    """Detect the language of the card based on field labels."""
    language_scores = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    
    for result in results:
        text = result[1][0]  # PaddleOCR format: [[[points]], [text, confidence]]
        prob = result[1][1]
        
        if prob < 0.4:
            continue
            
        text = text.strip()
        
        # Give higher weight to card titles which are strong language indicators
        if "CARTE EUROPEENNE" in text or "CARTE EUROPÉENNE" in text:
            language_scores['fr'] += 5  # Add 5 points for French title
            continue
            
        if "EUROPÄISCHE" in text:
            language_scores['de'] += 5  # Add 5 points for German title
            continue
            
        if "TESSERA EUROPEA" in text:
            language_scores['it'] += 5  # Add 5 points for Italian title
            continue
        
        # Check field labels
        for field in FIELD_LABELS:
            for lang in SUPPORTED_LANGUAGES:
                if any(label in text for label in FIELD_LABELS[field][lang]):
                    language_scores[lang] += 1
    
    # Get the language with the highest score
    detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]
    logger.debug(f"Detected language: {detected_lang} (scores: {language_scores})")
    return detected_lang

def extract_card_info(results):
    card_info = HealthCardInfo()
    
    # First detect the language
    detected_lang = detect_card_language(results)
    card_info.detected_language = detected_lang
    
    print(f"\nDetected language: {detected_lang}")
    print("\nDetected text:")
    for result in results:
        text = result[1][0]
        prob = result[1][1]
        print(f"{text} ({prob:.2%})")
    
    detected_values = {}
    potential_names = []
    
    for idx, result in enumerate(results):
        bbox = result[0]
        text = result[1][0]
        prob = result[1][1]
        
        text = text.strip()
        
        if prob < 0.4:
            continue

        if text in COUNTRY_CODES:
            continue
            
        # Simplified name detection - names are uppercase, without numbers, and have good confidence
        if text.isupper() and len(text) > 2 and prob > 0.7:
            # Check if text contains only letters and spaces
            if all(c.isalpha() or c.isspace() for c in text):
                if not any(text in label for labels in FIELD_LABELS.values() for label in labels[detected_lang]) and \
                   not any(word in text for word in EXCLUDED_WORDS[detected_lang]):
                    y_min = min(point[1] for point in bbox)
                    x_min = min(point[0] for point in bbox)
                    potential_names.append((text, y_min, x_min))
        
        # Universal personal number detection - "756.XXXX.XXXX.XX" is a standard Swiss format
        if "756" in text and prob > 0.7:
            # Clean the text to handle variations
            cleaned_text = text.strip()
            
            # Case 1: Already formatted with periods (e.g., "756.1234.5678.90")
            if cleaned_text.count('.') >= 2 and cleaned_text.startswith('756'):
                detected_values['personal_number'] = cleaned_text
            
            # Case 2: Just digits or missing periods (e.g., "7561234567890")
            else:
                digits = ''.join(filter(str.isdigit, cleaned_text))
                if digits.startswith("756") and len(digits) >= 13:
                    # Format it correctly with periods
                    formatted = digits[:3] + "." + digits[3:7] + "." + digits[7:11]
                    if len(digits) >= 13:
                        formatted += "." + digits[11:13]
                    detected_values['personal_number'] = formatted
        
        # Check field labels in detected language for insurance number
        if any(label in text for label in FIELD_LABELS['insurance_number'][detected_lang]):
            # Extract the number from this text or the next item
            number = ''.join(filter(str.isdigit, text))
            if number and len(number) >= 6:
                detected_values['insurance_number'] = number
            elif idx + 1 < len(results):
                next_text = results[idx + 1][1][0].strip()
                number = ''.join(filter(str.isdigit, next_text))
                if number and len(number) >= 6:
                    detected_values['insurance_number'] = number
        
        # Universal insurance code-name detection
        # This handles both combined format "0032 - Aquilana" and separate occurrences
        if '-' in text and prob > 0.6:
            parts = [p.strip() for p in text.split('-')]
            if len(parts) == 2:
                # First part should contain the insurance code (4-5 digits)
                code = ''.join(filter(str.isdigit, parts[0]))
                if len(code) >= 4 and len(code) <= 5:
                    detected_values['insurance_code'] = code
                    
                    # Second part is the insurance name
                    if parts[1]:
                        detected_values['insurance_name'] = parts[1].split()[0]  # Take first word
        
        # If we find a standalone numeric code that looks like an insurance code
        elif text.isdigit() and len(text) in (4, 5) and prob > 0.7:
            detected_values['insurance_code'] = text
            
            # Check if the next text might be the insurance provider name
            if idx + 1 < len(results):
                next_text = results[idx + 1][1][0].strip()
                next_prob = results[idx + 1][1][1]
                
                if next_prob > 0.7 and len(next_text) > 2 and next_text[0].isupper():
                    if not any(c.isdigit() for c in next_text):  # No digits in insurance name
                        detected_values['insurance_name'] = next_text.split()[0]
        
        # Card number detection - typically starts with "80756" followed by many digits
        if text.startswith("80756") and len(text) > 15 and prob > 0.7:
            detected_values['card_number'] = text
        
        # Date detection - finds both birth dates and expiry dates in DD/MM/YYYY format
        if len(text) == 10 and text.count("/") == 2:
            try:
                day, month, year = map(int, text.split("/"))
                # Basic date validation
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                    # Convert to datetime object for proper comparison
                    detected_date = datetime(year, month, day)
                    current_date = datetime.now()
                    
                    # If date is in the future, it's likely an expiry date
                    if detected_date > current_date:
                        detected_values['expiry_date'] = text
                    # If date is in the past, it's likely a birth date
                    else:
                        detected_values['birth_date'] = text
            except (ValueError, TypeError):
                # Skip invalid dates
                pass
    
    # Sort potential names by vertical (y) position first, then horizontal (x) position
    potential_names.sort(key=lambda x: (x[1], x[2]))
    
    # Filter out potential names that have already been assigned to other fields
    # particularly the insurance name
    filtered_names = []
    for name_tuple in potential_names:
        name_text = name_tuple[0]
        # Skip this text if it's already identified as insurance name or other non-name fields
        is_duplicate = False
        for field, value in detected_values.items():
            if field not in ['surname', 'first_name'] and value == name_text:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_names.append(name_tuple)
    
    if filtered_names:
        if len(filtered_names) >= 2:
            # If multiple names are detected, use a simple convention based on health card layouts:
            # The first name in order (higher on card) is the surname
            # The second name in order (lower on card) is the first name
            detected_values['surname'] = filtered_names[0][0]
            detected_values['first_name'] = ' '.join([name[0] for name in filtered_names[1:]])
            print(f"Names assigned: surname={detected_values['surname']}, first_name={detected_values['first_name']}")
        elif len(filtered_names) == 1:
            # If only one name is detected, assume it's the surname
            detected_values['surname'] = filtered_names[0][0]
            print(f"Single name detected: surname={detected_values['surname']}")
    
    for key, value in detected_values.items():
        print(f"- {key}: {value}")
        
    # Update card_info with detected values
    for field, value in detected_values.items():
        setattr(card_info, field, value)
    
    return card_info

def process_image_ocr(image):
    """
    Process an image through OCR and return the results.
    Args:
        image: numpy array of the image
    Returns:
        results: list of OCR results
    """
    enhanced = enhance_image(image)
    ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
    results = ocr.ocr(enhanced, cls=True)
    return results[0] if results else []

def process_single_image(image_path):
    try:
        output_dir = "detected_results"
        json_dir = "card_data"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        results = process_image_ocr(image)
        
        if not results:
            print(f"No text detected in {image_path}")
            return
        
        card_info = extract_card_info(results)
        
        json_filename = f"card_data_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(card_info.to_dict(), f, ensure_ascii=False, indent=2)
        
        annotated = create_annotated_image(image, results)
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated)
        
        print(f"\nExtracted Card Information for {os.path.basename(image_path)}:")
        # Custom printing order to ensure names are displayed first
        display_order = ['surname', 'first_name', 'birth_date', 'personal_number', 'insurance_code', 
                         'insurance_name', 'card_number', 'expiry_date', 'detected_language']
        card_dict = card_info.to_dict()
        for key in display_order:
            if card_dict.get(key):
                print(f"{key}: {card_dict[key]}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images(directory_path):
    output_dir = "detected_results"
    json_dir = "card_data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with Pool() as pool:
        pool.map(process_single_image, image_files)

if __name__ == "__main__":
    process_images("ids")
