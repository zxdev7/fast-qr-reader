from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import io
import os
import time
from io import BytesIO

# Try to import requests but don't fail if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uvicorn
import qrcode
from PIL import Image, ImageDraw, ImageFont
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

class QRCodePayload(BaseModel):
    """Request model for QR code generation"""
    payload: str
    overlay_text: Optional[str] = None
    overlay_color: Optional[str] = "#FFFF00"  # Default yellow color
    border_text: Optional[str] = None
    error_correction: Optional[int] = 1  # 0-3 for L, M, Q, H

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

def generate_qr_code(payload: str, error_correction: int = 1) -> Image.Image:
    """Generate a QR code from payload"""
    # Set error correction level (0-3 for L, M, Q, H)
    error_levels = [qrcode.constants.ERROR_CORRECT_L, 
                  qrcode.constants.ERROR_CORRECT_M,
                  qrcode.constants.ERROR_CORRECT_Q, 
                  qrcode.constants.ERROR_CORRECT_H]
    
    # Create QR code with minimal border (1 instead of default 4) and maximize box size
    qr = qrcode.QRCode(
        version=None,  # Auto-determine version based on payload size
        error_correction=error_levels[min(error_correction, 3)],
        box_size=20,  # Much larger box size to fill more of the frame
        border=1,     # Minimal border (1 is the minimum required for QR code standard)
    )
    
    qr.add_data(payload)
    qr.make(fit=True)
    
    # Create QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    
    return img

def add_overlay(qr_img: Image.Image, overlay_text: str, overlay_color: str = "#FFFF00", 
               border_text: str = None) -> Image.Image:
    """Add overlay to QR code image similar to example"""
    # Get QR code dimensions
    qr_width, qr_height = qr_img.size
    
    # Create a copy of the QR image
    final_img = qr_img.copy()
    
    # If overlay text is provided, add a colored rectangle with text
    if overlay_text:
        draw = ImageDraw.Draw(final_img)
        
        # Create much smaller overlay rectangle in the center
        rect_width = int(qr_width * 0.4)  # 40% of QR width (reduced from 50%)
        rect_height = int(qr_height * 0.14)  # 14% of QR height (reduced from 18%)
        
        # Position the rectangle in the center
        left = (qr_width - rect_width) // 2
        top = (qr_height - rect_height) // 2
        right = left + rect_width
        bottom = top + rect_height
        
        # Parse overlay color to add transparency (50% opacity)
        # If overlay_color is a hex string, convert to RGBA with 50% opacity
        if isinstance(overlay_color, str) and overlay_color.startswith('#'):
            # Default to yellow with 50% opacity if parsing fails
            try:
                # Convert hex to RGB
                r = int(overlay_color[1:3], 16) if len(overlay_color) >= 3 else 255
                g = int(overlay_color[3:5], 16) if len(overlay_color) >= 5 else 255
                b = int(overlay_color[5:7], 16) if len(overlay_color) >= 7 else 0
                # Use yellow with 70% opacity (77) for better QR visibility
                overlay_color = (r, g, b, 77)
            except ValueError:
                overlay_color = (255, 255, 0, 77)  # Yellow with 70% opacity
        else:
            # Default to yellow with 70% opacity
            overlay_color = (255, 255, 0, 77)
            
        # Convert to RGBA if needed for transparency support
        if final_img.mode != 'RGBA':
            final_img = final_img.convert('RGBA')
            
        # Draw the colored rectangle with transparency
        draw = ImageDraw.Draw(final_img, 'RGBA')
        draw.rectangle([left, top, right, bottom], fill=overlay_color)
        
        # Function to wrap text to fit within width
        def wrap_text(text, font, max_width):
            words = text.split(' ')
            lines = []
            line = []
            current_width = 0

            # Special handling for Thai text (which often doesn't use spaces)
            # If no spaces in text, try to break by character groups (roughly)
            if len(words) == 1 and len(text) > 15:
                # For Thai text without spaces, break into roughly equal chunks
                char_per_line = max(5, len(text) // 2)  # At least 5 chars per line, or divide by 2
                idx = 0
                while idx < len(text):
                    lines.append(text[idx:idx+char_per_line])
                    idx += char_per_line
                return lines
                
            # For normal text with spaces
            for word in words:
                word_width = font.getlength(word + ' ')
                if current_width + word_width <= max_width:
                    line.append(word)
                    current_width += word_width
                else:
                    lines.append(' '.join(line))
                    line = [word]
                    current_width = font.getlength(word + ' ')
            
            if line:
                lines.append(' '.join(line))
                
            return lines
        
        # Find best size for text - even smaller than before
        font_size = int(rect_height / 4.0)  # Reduced from 3.5 to 4.0 for even smaller text
        font = None
        
        # Try to find and load a Thai-compatible font
        thai_font_paths = [
            "https://cdn.jsdelivr.net/gh/lazywasabi/thai-web-fonts@7/fonts/Sarabun/Sarabun-Light.woff2"
        ]
        
        # Try each font path until a valid one is found
        for font_path in thai_font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except Exception as e:
                print(f"Error loading font {font_path}: {e}")
        
        # If no font was found, use default
        if not font:
            font = ImageFont.load_default()
            font_size = 12  # Even smaller default size
        
        # Wrap text to fit in rectangle
        wrapped_text = wrap_text(overlay_text, font, rect_width - 10)  # 5px padding on each side
        
        # Calculate total text height
        line_height = font_size + 2  # Add a small gap between lines
        total_text_height = len(wrapped_text) * line_height
        
        # Draw each line of text
        y_offset = top + (rect_height - total_text_height) // 2  # Center vertically
        
        for line in wrapped_text:
            # Get line width for horizontal centering
            text_bbox = font.getbbox(line)
            text_width = text_bbox[2] - text_bbox[0]
            
            # Calculate position to center this line
            text_x = left + (rect_width - text_width) // 2
            
            # Draw the text in red color
            draw.text((text_x, y_offset), line, fill="#FF0000", font=font)
            y_offset += line_height  # Move to next line
    
    # If border text is provided, add text around the border
    if border_text:
        # Create a canvas with minimal padding and transparent background
        padding = 30  # Reduced padding so text overlaps with QR code slightly
        canvas = Image.new('RGBA', (qr_width + padding*2, qr_height + padding*2), (255, 255, 255, 0))
        canvas.paste(final_img, (padding, padding))
        
        draw = ImageDraw.Draw(canvas)
        
        # Function to load a font from local path or URL
        def load_font(path, size):
            try:
                if path.startswith('http'):
                    if HAS_REQUESTS:
                        # Download font from URL if requests is available
                        try:
                            response = requests.get(path)
                            if response.status_code == 200:
                                # Load font from the downloaded content
                                return ImageFont.truetype(BytesIO(response.content), size=size)
                            else:
                                print(f"Failed to download font: {response.status_code}")
                        except Exception as e:
                            print(f"Error downloading font: {str(e)}")
                    # If requests failed or not available, fall back to default
                    print("Using default font because online font could not be loaded")
                    return ImageFont.load_default()
                else:
                    # Load font from local file
                    return ImageFont.truetype(path, size=size)
            except Exception as e:
                print(f"Error loading font {path}: {str(e)}")
                return None
        
        # Function to find optimal font size for a given width
        def get_optimal_font_size(text, available_width, font_path, min_size=12, max_size=50):
            # Try each font size from largest to smallest
            for size in range(max_size, min_size-1, -1):
                test_font = load_font(font_path, size)
                if test_font:
                    try:
                        text_width = draw.textlength(text, font=test_font)
                        if text_width <= available_width * 0.95:  # 95% of available width
                            return size, test_font
                    except Exception as e:
                        print(f"Font size error: {str(e)}")
                        continue
            
            # If we couldn't find a good size, return the minimum size or default
            min_font = load_font(font_path, min_size)
            if min_font:
                return min_size, min_font
            else:
                return min_size, ImageFont.load_default()
        
        # Try to find a font that supports Thai characters for border text
        thai_fonts = [
            # Include both online and local Thai fonts for better compatibility
            "https://cdn.jsdelivr.net/gh/lazywasabi/thai-web-fonts@7/fonts/Sarabun/Sarabun-Light.woff2"
        ]
        
        # Use first available font (preferring URLs for Thai support)
        font_path = thai_fonts[0] if thai_fonts else None
        
        # Calculate positions and dimensions
        canvas_width, canvas_height = canvas.size
        
        # Available width is the full canvas width
        horizontal_font_size, horizontal_font = get_optimal_font_size(
            border_text, canvas_width - padding/2, font_path, max_size=40)
        text_width = draw.textlength(border_text, font=horizontal_font)
        top_x = int((canvas_width - text_width) / 2)  # Center text
        top_y = int(padding//3)  # Move further inside to overlap QR code more
        draw.text((top_x, top_y), border_text, fill="#FF0000", font=horizontal_font)
        
        # Bottom border text removed as requested
        
        # Calculate vertical font size for sides
        vertical_font_size, vertical_font = get_optimal_font_size(
            border_text, canvas_height - padding/2, font_path, max_size=35)
            
        # Left border - horizontal baseline with rotated text (reading bottom to top)
        # Calculate size first
        font_height = vertical_font.getbbox("Áp")[3]  # Get approx font height with accents
        text_width = int(draw.textlength(border_text, font=vertical_font))
        
        # Create a new image for the text (in normal horizontal orientation)
        text_img_width = text_width + 20  # Add padding
        text_img_height = font_height + 20  # Add padding
        
        # Ensure dimensions are integers
        left_img = Image.new('RGBA', 
                            (int(text_img_width), int(text_img_height)), 
                            (255, 255, 255, 0))
        left_draw = ImageDraw.Draw(left_img)
        
        # Draw text horizontally (normal orientation)
        left_draw.text((10, 10), border_text, fill="#FF0000", font=vertical_font)
        
        # Rotate the image for the left border (to read bottom to top) - 90 degrees counter-clockwise
        left_img = left_img.rotate(90, expand=True)
        
        # Paste onto the main canvas - more overlapping onto QR code
        left_x = int(padding//2)  # Move further inside to overlap QR code more
        left_y = int((canvas_height - left_img.height) / 2)  # Center vertically
        canvas.paste(left_img, (left_x, left_y), left_img)
        
        # Right border - horizontal baseline with rotated text (reading top to bottom)
        # Calculate size first
        font_height = vertical_font.getbbox("Áp")[3]  # Get approx font height with accents
        text_width = int(draw.textlength(border_text, font=vertical_font))
        
        # Create a new image for the text (in normal horizontal orientation)
        text_img_width = text_width + 20  # Add padding
        text_img_height = font_height + 20  # Add padding
        
        # Ensure dimensions are integers
        right_img = Image.new('RGBA', 
                             (int(text_img_width), int(text_img_height)), 
                             (255, 255, 255, 0))
        right_draw = ImageDraw.Draw(right_img)
        
        # Draw text horizontally (normal orientation)
        right_draw.text((10, 10), border_text, fill="#FF0000", font=vertical_font)
        
        # Rotate the image for the right border (to read top to bottom) - 270 degrees (or -90 degrees)
        right_img = right_img.rotate(270, expand=True)
        
        # Paste onto the main canvas - more overlapping onto QR code
        right_x = int(canvas_width - padding//2 - right_img.width)  # Move further inside to overlap QR code more
        right_y = int((canvas_height - right_img.height) / 2)  # Center vertically
        canvas.paste(right_img, (right_x, right_y), right_img)
        
        return canvas
    
    return final_img

@app.post("/generate-qr")
async def generate_qr(request: QRCodePayload):  # Temporarily disabled token verification
    """
    Generate QR code from payload
    
    - **payload**: Text/URL to encode in QR code
    - **overlay_text**: Optional text to display in center overlay
    - **overlay_color**: Color for overlay (hex format, default: yellow)
    - **border_text**: Optional text to display around border
    - **error_correction**: Error correction level (0-3 for L, M, Q, H)
    
    Returns:
        JSON with base64 encoded PNG image
    """
    start_time = time.time()
    
    try:
        # Generate QR code
        qr_img = generate_qr_code(
            request.payload, 
            error_correction=request.error_correction
        )
        
        # Add overlay if requested
        if request.overlay_text or request.border_text:
            qr_img = add_overlay(
                qr_img, 
                request.overlay_text, 
                request.overlay_color,
                request.border_text
            )
        
        # Convert to PNG and encode as base64
        buffered = io.BytesIO()
        qr_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare response
        response = {
            "qr_code": img_base64,
            "payload": request.payload,
            "processing_time": time.time() - start_time
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating QR code: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
