# OCR Application

This application uses PaddleOCR to process health insurance cards.

# Swiss Health Insurance Card OCR Reader

This script is an OCR (Optical Character Recognition) application designed to automatically read information from Swiss health insurance cards.

## Features

- Automatic extraction of all essential card information
- Image preprocessing and enhancement
- JSON format data output
- Annotated image output
- High accuracy rate
- REST API endpoint for processing cards

## Extracted Information

- Versicherten-Nr (Insurance Number)
- Name (Last Name)
- Vorname (First Name)
- Geburtsdatum (Date of Birth)
- Kennnummer (Personal Number)
- Karte-Nr (Card Number)
- Ablaufdatum (Expiry Date)
- Versicherung (Insurance Company)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### As a Script

1. Create a folder named 'ids' on root
2. Copy the card photos you want to process into this folder
3. Run the script:

```bash
python ocr_reader.py
```

### As an API

1. Start the API server:

```bash
python api.py
```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:
   - GET `/`: Welcome message
   - POST `/process-card/`: Process a health insurance card image
4. Example API usage with curl:

```bash
curl -X POST "http://localhost:8000/process-card/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/card/image.jpg"
```

5. Example API usage with Python requests:

```python
import requests

url = "http://localhost:8000/process-card/"
files = {"file": open("path/to/your/card/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

6. API Response Format:

```json
{
  "status": "success",
  "card_info": {
    "insurance_number": "1234567",
    "surname": "MUSTERMANN",
    "first_name": "MAX",
    "birth_date": "01/01/1990",
    "personal_number": "123.4567.8901.23",
    "insurance_provider_id": "0032 - Sample Insurance",
    "card_number": "80756000320001234567",
    "expiry_date": "31/12/2025"
  },
  "confidence_scores": {
    "MUSTERMANN": "95%",
    "MAX": "92%"
    // ... other detected text and their confidence scores
  },
  "images": {
    "original": "base64_encoded_original_image_string",
    "annotated": "base64_encoded_annotated_image_string"
  },
  "processing_details": {
    "original_size": "1200x800",
    "processed_size": "1000x667",
    "file_name": "health_card.jpg",
    "content_type": "image/jpeg"
  }
}
```

7. Working with Base64 Images:

```python
# Converting base64 back to image in Python
import base64
import cv2
import numpy as np

def base64_to_image(base64_string):
    # Remove data:image/jpeg;base64, if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64 string
    img_data = base64.b64decode(base64_string)

    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# Save base64 image
def save_base64_image(base64_string, output_path):
    image = base64_to_image(base64_string)
    cv2.imwrite(output_path, image)
```

## Outputs

The script generates two types of outputs:

1. **JSON Files** (in `card_data` directory)

   - Separate JSON file for each card
   - All extracted information in structured format

2. **Annotated Images** (in `detected_results` directory)
   - Images with visually marked detected areas
   - Detected text and confidence scores


## Debugging

- Displays all detected text and confidence scores during execution
- Provides detailed error messages
- Shows warnings for unprocessable images

## Notes

- Supported image formats: .jpg, .jpeg, .png
- Minimum confidence threshold: 0.5 (50%)
- Recommended image resolution: maximum 1500px (automatically scaled)

## Performance Optimization

- Efficient image processing pipeline
- Memory-optimized operations
- Fast text detection and recognition

## Error Handling

- Robust error handling for various scenarios
- Graceful handling of malformed or low-quality images
- Comprehensive logging of processing steps

## Best Practices for Usage

1. **Image Quality**

   - Ensure good lighting when taking photos
   - Avoid glare on the card surface
   - Keep the card aligned horizontally

2. **File Management**

   - Use meaningful filenames
   - Regular backup of output files
   - Clean up old processed files

3. **System Requirements**
   - Minimum 4GB RAM recommended
   - Python 3.7 or higher
   - Sufficient disk space for output files

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
