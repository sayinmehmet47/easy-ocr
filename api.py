from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from ocr_reader import extract_card_info, process_image_ocr
from utils import (
    encode_image_to_base64,
    create_annotated_image,
    SUPPORTED_LANGUAGES,
    logger
)

app = FastAPI(
    title="Health Insurance Card OCR API",
    description="API for extracting information from German, French, and Italian health insurance cards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {
        "message": "Health Insurance Card OCR API",
        "supported_languages": SUPPORTED_LANGUAGES
    }

@app.post("/process-card/")
async def process_card(
    file: UploadFile = File(description="Health insurance card image file")
):
    try:
        logger.debug(f"Received request for file processing")
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
            
        logger.debug(f"Received file: {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store original image as base64
        original_image_base64 = encode_image_to_base64(image)
        
        # Process image with OCR
        results = process_image_ocr(image)
        
        if not results:
            return JSONResponse(
                status_code=422,
                content={"error": "No text detected in the image"}
            )
        
        # Extract card information
        card_info = extract_card_info(results)
        
        # Create annotated image
        annotated = create_annotated_image(image, results)
        annotated_image_base64 = encode_image_to_base64(annotated)

        # Save images to disk
        # try:
        #     save_dir = 'detected_images'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)

        #     # Save original image
        #     original_path = os.path.join(save_dir, 'original_image.jpg')
        #     success = cv2.imwrite(original_path, image)
        #     logger.debug(f"Saving original image: {'Success' if success else 'Failed'} - Path: {original_path}")

        #     # Save annotated image
        #     annotated_path = os.path.join(save_dir, 'annotated_image.jpg')
        #     success = cv2.imwrite(annotated_path, annotated)
        #     logger.debug(f"Saving annotated image: {'Success' if success else 'Failed'} - Path: {annotated_path}")
            
        #     logger.info(f"Images saved to {save_dir}/ directory")
            
        # except Exception as save_error:
        #     logger.error(f"Error saving images: {str(save_error)}", exc_info=True)
            
        # Return the extracted information with images
        response_data = {
            "status": "success",
            "card_info": card_info.to_dict(),
            "confidence_scores": {
                result[1][0]: f"{result[1][1]:.2%}" for result in results if result[1][1] > 0.5
            },
            "images": {
                "original": original_image_base64,
                "annotated": annotated_image_base64
            },
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the image: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 
