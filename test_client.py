import requests
import base64
import argparse
import os
import json
from typing import Optional, Dict, Any

class QRCodeAPIClient:
    """Test client for QR Code Reader API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: Optional[str] = None):
        self.base_url = base_url
        self.headers = {}
        
        # Set authorization header if token is provided
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        return self._handle_response(response)
    
    def read_qr_file(self, file_path: str, detailed: bool = False) -> Dict[str, Any]:
        """Read QR code from image file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
            params = {"detailed": "true" if detailed else "false"}
            response = requests.post(
                f"{self.base_url}/read-qr", 
                files=files,
                params=params,
                headers=self.headers
            )
        
        return self._handle_response(response)
    
    def read_qr_base64(self, file_path: str, detailed: bool = False) -> Dict[str, Any]:
        """Read QR code from base64 encoded image"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
        
        payload = {
            "image": base64_data,
            "detailed": detailed
        }
        
        response = requests.post(
            f"{self.base_url}/read-qr-base64",
            json=payload,
            headers=self.headers
        )
        
        return self._handle_response(response)
    
    def read_qr_advanced(self, file_path: str) -> Dict[str, Any]:
        """Read QR code using advanced detection methods"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
            response = requests.post(
                f"{self.base_url}/read-qr-advanced", 
                files=files,
                headers=self.headers
            )
        
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e}"
            try:
                error_detail = response.json()
                error_msg = f"{error_msg} - {error_detail.get('detail', '')}"
            except:
                pass
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request Error: {e}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {response.text}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="QR Code Reader API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--token", help="API token for authorization")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--mode", choices=["standard", "base64", "advanced"], default="standard", 
                       help="API mode to use (standard, base64, or advanced)")
    parser.add_argument("--detailed", action="store_true", help="Use detailed mode for standard/base64 API")
    
    args = parser.parse_args()
    
    client = QRCodeAPIClient(args.url, args.token)
    
    try:
        # Check health first
        health = client.health_check()
        print(f"API Health: {health['status']}")
        print(f"API Version: {health['version']}")
        print(f"OpenCV Version: {health['opencv_version']}")
        print("-" * 40)
        
        # Process image based on mode
        if args.mode == "standard":
            result = client.read_qr_file(args.image, args.detailed)
        elif args.mode == "base64":
            result = client.read_qr_base64(args.image, args.detailed)
        elif args.mode == "advanced":
            result = client.read_qr_advanced(args.image)
        
        # Print results
        print(f"QR Code Detection Results:")
        print(f"Processing Time: {result['processing_time']:.4f} seconds")
        print(f"Detailed Mode: {'Yes' if result['detailed_mode'] else 'No'}")
        print(f"QR Codes Found: {result['count']}")
        
        if result['count'] > 0:
            for i, qr in enumerate(result['results']):
                print(f"\nQR Code #{i+1}:")
                print(f"  Data: {qr.get('data')}")
                print(f"  Method: {qr.get('method')}")
                if 'preprocessing' in qr:
                    print(f"  Preprocessing: {qr.get('preprocessing')}")
                if 'processing_time' in qr:
                    print(f"  Processing Time: {qr.get('processing_time'):.4f} seconds")
        else:
            print("\nNo QR codes detected in the image.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
