import cv2
import numpy as np
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Multi-language field labels
FIELD_LABELS = {
    'insurance_number': {
        'de': ['Versicherten-Nr', 'Versicherten-Nr.', 'ersicherten-Nr', 'ersicherten-Nr.'],
        'fr': ['N° d\'assuré', 'Numéro d\'assuré', 'N° assuré', 'No. d\'assuré'],
        'it': ['N. assicurato', 'Numero assicurato', 'N° assicurato']
    },
    'surname': {
        'de': ['3. Name', 'Name', '3.Name'],
        'fr': ['3. Nom', 'Nom', '3.Nom'],
        'it': ['3. Cognome', 'Cognome', '3.Cognome']
    },
    'first_name': {
        'de': ['4. Vornamen', 'Vornamen', '4.Vornamen'],
        'fr': ['4. Prénoms', 'Prénoms', '4.Prénoms'],
        'it': ['4. Nome', 'Nome', '4.Nome']
    },
    'birth_date': {
        'de': ['5. Geburtsdatum', 'Geburtsdatum', '5.Geburtsdatum', 'Geburtsdat'],
        'fr': ['5. Date de naissance', 'Date de naissance', '5.Date de naissance'],
        'it': ['5. Data di nascita', 'Data di nascita', '5.Data di nascita']
    },
    'personal_number': {
        'de': ['6. Persönliche Kennnummer', 'Kennnummer', '6. Personliche'],
        'fr': ['6. Numéro personnel', 'Numéro personnel', '6.Numéro personnel'],
        'it': ['6. Numero personale', 'Numero personale', '6.Numero personale']
    },
    'insurance_provider_id': {
        'de': ['7. Kennnummer des Trägers', '7. Kennnummer'],
        'fr': ['7. Code de l\'organisme', '7.Code organisme'],
        'it': ['7. Codice ente', '7.Codice ente']
    },
    'card_number': {
        'de': ['8. Kennnummer der Karte'],
        'fr': ['8. Numéro de la carte'],
        'it': ['8. Numero della carta']
    },
    'expiry_date': {
        'de': ['9. Ablaufdatum'],
        'fr': ['9. Date d\'expiration'],
        'it': ['9. Data di scadenza']
    }
}

# Common words to exclude from name detection
EXCLUDED_WORDS = {
    'de': ['KARTE', 'VERSICHERUNG', 'EUROPEAN', 'EUROPÄISCHE', 'EUROPAISCHE'],
    'fr': ['CARTE', 'ASSURANCE', 'EUROPÉENNE', 'SANTÉ'],
    'it': ['CARTA', 'ASSICURAZIONE', 'EUROPEA', 'SANITARIA']
}

COUNTRY_CODES = ['CH']

SUPPORTED_LANGUAGES = ['de', 'fr', 'it']

def encode_image_to_base64(image):
    """Convert an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')
    
def enhance_image(image):
    """Enhance image for better OCR processing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
    return enhanced

def create_annotated_image(image, results):
    """Create annotated image with detected text regions."""
    annotated = image.copy()
    for result in results:
        points = np.array(result[0]).astype(np.int32)
        text = result[1][0]
        prob = result[1][1]
        
        if prob > 0.5:
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
            x, y = points[0]
            cv2.putText(annotated, f"{text}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated 
