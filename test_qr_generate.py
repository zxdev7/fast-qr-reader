import requests
import json
import base64
from PIL import Image
import io
import sys
import os

# URL of the QR code generator API
API_URL = "http://localhost:8000/generate-qr"

def save_qr_image(base64_str, filename="generated_qr.png"):
    """Save base64 encoded image to file"""
    try:
        img_data = base64.b64decode(base64_str)
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"QR code saved as {filename}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def generate_qr_code(payload, overlay_text=None, overlay_color="#FFFF00", border_text=None, error_correction=1):
    """Generate QR code with optional overlay and border text"""
    
    # Prepare request data
    request_data = {
        "payload": payload,
        "error_correction": error_correction
    }
    
    # Add optional parameters if provided
    if overlay_text:
        request_data["overlay_text"] = overlay_text
    
    if overlay_color:
        request_data["overlay_color"] = overlay_color
    
    if border_text:
        request_data["border_text"] = border_text
    
    try:
        # Send POST request to API
        response = requests.post(API_URL, json=request_data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Save QR code image
            if "qr_code" in result:
                output_filename = f"qr_code_{payload.replace('://', '_').replace('/', '_').replace('.', '_')}.png"
                save_qr_image(result["qr_code"], output_filename)
                
                print(f"QR code generated successfully!")
                print(f"Processing time: {result.get('processing_time', 'N/A')} seconds")
                return output_filename
            else:
                print("Error: No QR code in response")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")
    
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_qr_generate.py <payload> [overlay_text] [border_text]")
        print("Example: python test_qr_generate.py 'https://example.com' 'SCAN ME' 'OFFICIAL QR CODE'")
        sys.exit(1)
    
    payload = sys.argv[1]
    overlay_text = sys.argv[2] if len(sys.argv) > 2 else "คิวอาร์โค้ดนี้ใช้สำหรับ"
    border_text = sys.argv[3] if len(sys.argv) > 3 else "www.p2wtqpup.com เท่านั้น"
    
    print(f"Generating QR code for: {payload}")
    print(f"With overlay text: {overlay_text}")
    print(f"With border text: {border_text}")
    
    filename = generate_qr_code(payload, overlay_text, "#FFFF00", border_text)
    
    if filename and os.path.exists(filename):
        print(f"QR code saved as: {filename}")
    else:
        print("Failed to generate QR code.")
