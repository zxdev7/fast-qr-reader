from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import time
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(
    title="QR Code Reader API",
    description="API for reading QR codes from images with multiple detection methods",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token verification
def verify_token(authorization: Optional[str] = Header(None)):
    """Verify the API token from the Authorization header"""
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        return True  # If no token is set in environment, skip verification
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if authorization != f"Bearer {api_token}":
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    return True

class Base64Image(BaseModel):
    """Request model for base64 encoded image"""
    image: str
    detailed: bool = False

def read_qr_opencv(image: np.ndarray) -> List[Dict[str, Any]]:
    """Read QR codes using OpenCV QRCodeDetector"""
    start_time = time.time()
    qr_detector = cv2.QRCodeDetector()
    
    # Try to detect and decode
    data, bbox, _ = qr_detector.detectAndDecode(image)
    
    results = []
    if data:
        results.append({
            "data": data,
            "bounding_box": bbox.tolist() if bbox is not None else None,
            "method": "opencv",
            "processing_time": time.time() - start_time
        })
    
    return results

def read_qr_wechat(image: np.ndarray) -> List[Dict[str, Any]]:
    """Read QR codes using OpenCV's WeChatQRCode detector (more robust than standard QRCodeDetector)"""
    start_time = time.time()
    
    try:
        # Try to use WeChatQRCode detector if available
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        data, points = wechat_detector.detectAndDecode(image)
        
        results = []
        for i, d in enumerate(data):
            if d:
                bbox = points[i].tolist() if i < len(points) else None
                results.append({
                    "data": d,
                    "bounding_box": bbox,
                    "method": "wechat_qrcode",
                    "processing_time": time.time() - start_time
                })
        return results
    except (AttributeError, cv2.error):
        # WeChatQRCode not available in this OpenCV build
        return []

def preprocess_image(image: np.ndarray, method: str = "default") -> np.ndarray:
    """Apply preprocessing to the image based on the specified method"""
    if method == "default":
        return image
    
    if method == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    
    if method == "adaptive_threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    if method == "enhance_contrast":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    
    # Default fallback
    return image

def advanced_qr_detection(image: np.ndarray) -> List[Dict[str, Any]]:
    """Advanced QR detection with multiple preprocessing methods and techniques"""
    start_time = time.time()
    results = []
    
    # Try different preprocessing methods
    preprocessing_methods = ["default", "grayscale", "threshold", "adaptive_threshold", "enhance_contrast"]
    
    for method in preprocessing_methods:
        processed_image = preprocess_image(image, method)
        
        # Try OpenCV detection
        opencv_results = read_qr_opencv(processed_image)
        for result in opencv_results:
            result["preprocessing"] = method
            results.append(result)
            
            # Early termination if we found something
            if result["data"]:
                return results
        
        # Try WeChatQRCode detection if available
        wechat_results = read_qr_wechat(processed_image)
        for result in wechat_results:
            result["preprocessing"] = method
            results.append(result)
            
            # Early termination if we found something
            if result["data"]:
                return results
    
    # If no results yet, try rotating the image
    if not results:
        angles = [90, 180, 270]
        for angle in angles:
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # Try OpenCV detection on rotated image
            opencv_results = read_qr_opencv(rotated)
            for result in opencv_results:
                result["preprocessing"] = f"rotation_{angle}"
                results.append(result)
                
                # Early termination if we found something
                if result["data"]:
                    return results
            
            # Try WeChatQRCode detection on rotated image
            wechat_results = read_qr_wechat(rotated)
            for result in wechat_results:
                result["preprocessing"] = f"rotation_{angle}"
                results.append(result)
                
                # Early termination if we found something
                if result["data"]:
                    return results
    
    # Add total processing time
    total_time = time.time() - start_time
    for result in results:
        result["total_processing_time"] = total_time
    
    return results

@app.get("/health")
async def health_check(token_verified: bool = Depends(verify_token)):
    """Health check endpoint with system information"""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": app.version,
        "opencv_version": cv2.__version__,
    }

@app.post("/read-qr")
async def read_qr(file: UploadFile = File(...), detailed: bool = False, token_verified: bool = Depends(verify_token)):
    """
    Read QR code from uploaded image file
    
    - **file**: The image file containing QR code
    - **detailed**: If True, use advanced detection methods (slower but more accurate)
    """
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read the file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Process the image
    if detailed:
        results = advanced_qr_detection(image)
    else:
        # Try OpenCV first (faster)
        results = read_qr_opencv(image)
        
        # If no results, try WeChatQRCode if available
        if not results:
            results = read_qr_wechat(image)
    
    # Prepare response
    response = {
        "results": results,
        "count": len(results),
        "processing_time": time.time() - start_time,
        "detailed_mode": detailed
    }
    
    return response

@app.post("/read-qr-base64")
async def read_qr_base64(request: Base64Image, token_verified: bool = Depends(verify_token)):
    """
    Read QR code from base64 encoded image
    
    - **image**: Base64 encoded image string
    - **detailed**: If True, use advanced detection methods (slower but more accurate)
    """
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        
        # Process the image
        if request.detailed:
            results = advanced_qr_detection(image)
        else:
            # Try OpenCV first (faster)
            results = read_qr_opencv(image)
            
            # If no results, try WeChatQRCode if available
            if not results:
                results = read_qr_wechat(image)
        
        # Prepare response
        response = {
            "results": results,
            "count": len(results),
            "processing_time": time.time() - start_time,
            "detailed_mode": request.detailed
        }
        
        return response
    
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

@app.post("/read-qr-advanced")
async def read_qr_advanced(file: UploadFile = File(...), token_verified: bool = Depends(verify_token)):
    """
    Read QR code using advanced detection methods (slower but more accurate)
    
    - **file**: The image file containing QR code
    """
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read the file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Process the image with advanced detection
    results = advanced_qr_detection(image)
    
    # Prepare response
    response = {
        "results": results,
        "count": len(results),
        "processing_time": time.time() - start_time,
        "detailed_mode": True
    }
    
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
