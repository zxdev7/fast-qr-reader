import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter

class AdvancedQRProcessor:
    """Advanced QR code processing with multiple techniques and optimizations"""
    
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Main detection method that tries multiple techniques"""
        start_time = time.time()
        results = []
        
        # First try standard methods
        basic_results = self._try_basic_detection(image)
        if basic_results:
            for result in basic_results:
                result["total_processing_time"] = time.time() - start_time
            return basic_results
        
        # If no results, try preprocessing techniques
        preprocessed_results = self._try_preprocessed_detection(image)
        if preprocessed_results:
            for result in preprocessed_results:
                result["total_processing_time"] = time.time() - start_time
            return preprocessed_results
        
        # If still no results, try advanced techniques
        advanced_results = self._try_advanced_detection(image)
        if advanced_results:
            for result in advanced_results:
                result["total_processing_time"] = time.time() - start_time
            return advanced_results
        
        # If no QR codes found, return empty list with processing time
        return [{
            "data": None,
            "method": "all_methods",
            "preprocessing": "all_techniques",
            "error": "No QR codes detected",
            "total_processing_time": time.time() - start_time
        }]
    
    def _try_basic_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try basic detection methods"""
        results = []
        
        # Try OpenCV QRCodeDetector
        opencv_start = time.time()
        data, bbox, _ = self.qr_detector.detectAndDecode(image)
        if data:
            results.append({
                "data": data,
                "bounding_box": bbox.tolist() if bbox is not None else None,
                "method": "opencv",
                "preprocessing": "none",
                "processing_time": time.time() - opencv_start
            })
            return results
        
        # Try WeChatQRCode detector if available
        try:
            wechat_start = time.time()
            wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
            data, points = wechat_detector.detectAndDecode(image)
            
            for i, d in enumerate(data):
                if d:
                    bbox = points[i].tolist() if i < len(points) else None
                    results.append({
                        "data": d,
                        "bounding_box": bbox,
                        "method": "wechat_qrcode",
                        "preprocessing": "none",
                        "processing_time": time.time() - wechat_start
                    })
                    return results
        except (AttributeError, cv2.error):
            # WeChatQRCode not available in this OpenCV build
            pass
            
        return results
    
    def _try_preprocessed_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try detection with various preprocessing techniques"""
        results = []
        
        # Define preprocessing methods
        preprocessing_methods = [
            ("grayscale", self._preprocess_grayscale),
            ("threshold", self._preprocess_threshold),
            ("adaptive_threshold", self._preprocess_adaptive_threshold),
            ("enhance_contrast", self._preprocess_enhance_contrast),
            ("sharpen", self._preprocess_sharpen)
        ]
        
        # Try each preprocessing method
        for name, preprocess_func in preprocessing_methods:
            processed = preprocess_func(image)
            
            # Try OpenCV QRCodeDetector
            opencv_start = time.time()
            data, bbox, _ = self.qr_detector.detectAndDecode(processed)
            if data:
                results.append({
                    "data": data,
                    "bounding_box": bbox.tolist() if bbox is not None else None,
                    "method": "opencv",
                    "preprocessing": name,
                    "processing_time": time.time() - opencv_start
                })
                return results
            
            # Try WeChatQRCode detector if available
            try:
                wechat_start = time.time()
                wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                data, points = wechat_detector.detectAndDecode(processed)
                
                for i, d in enumerate(data):
                    if d:
                        bbox = points[i].tolist() if i < len(points) else None
                        results.append({
                            "data": d,
                            "bounding_box": bbox,
                            "method": "wechat_qrcode",
                            "preprocessing": name,
                            "processing_time": time.time() - wechat_start
                        })
                        return results
            except (AttributeError, cv2.error):
                # WeChatQRCode not available in this OpenCV build
                pass
        
        return results
    
    def _try_advanced_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try advanced detection techniques"""
        results = []
        
        # Try rotations
        rotation_results = self._try_rotations(image)
        if rotation_results:
            return rotation_results
        
        # Try region-based detection
        region_results = self._try_region_detection(image)
        if region_results:
            return region_results
        
        # Try PIL-based enhancements
        pil_results = self._try_pil_enhancements(image)
        if pil_results:
            return pil_results
        
        return results
    
    def _try_rotations(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try detecting QR codes in rotated images"""
        results = []
        angles = [90, 180, 270]
        
        for angle in angles:
            # Rotate image
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # Try OpenCV QRCodeDetector
            opencv_start = time.time()
            data, bbox, _ = self.qr_detector.detectAndDecode(rotated)
            if data:
                results.append({
                    "data": data,
                    "bounding_box": bbox.tolist() if bbox is not None else None,
                    "method": "opencv",
                    "preprocessing": f"rotation_{angle}",
                    "processing_time": time.time() - opencv_start
                })
                return results
            
            # Try WeChatQRCode detector if available
            try:
                wechat_start = time.time()
                wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                data, points = wechat_detector.detectAndDecode(rotated)
                
                for i, d in enumerate(data):
                    if d:
                        bbox = points[i].tolist() if i < len(points) else None
                        results.append({
                            "data": d,
                            "bounding_box": bbox,
                            "method": "wechat_qrcode",
                            "preprocessing": f"rotation_{angle}",
                            "processing_time": time.time() - wechat_start
                        })
                        return results
            except (AttributeError, cv2.error):
                # WeChatQRCode not available in this OpenCV build
                pass
        
        return results
    
    def _try_region_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try detecting QR codes in specific regions using contour detection"""
        results = []
        
        # Convert to grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = (image.shape[0] * image.shape[1]) * 0.01  # 1% of image area
        max_area = (image.shape[0] * image.shape[1]) * 0.9   # 90% of image area
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        
        # Sort contours by area (largest first)
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Process each contour region
        for i, contour in enumerate(filtered_contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region
            region = image[y:y+h, x:x+w]
            
            # Skip if region is too small
            if region.size == 0 or w < 20 or h < 20:
                continue
            
            # Try OpenCV QRCodeDetector
            opencv_start = time.time()
            data, bbox, _ = self.qr_detector.detectAndDecode(region)
            if data:
                # Adjust bounding box coordinates to original image
                if bbox is not None:
                    bbox = bbox + np.array([x, y])
                
                results.append({
                    "data": data,
                    "bounding_box": bbox.tolist() if bbox is not None else None,
                    "method": "opencv",
                    "preprocessing": f"region_{i}",
                    "processing_time": time.time() - opencv_start
                })
                return results
            
            # Try WeChatQRCode detector if available
            try:
                wechat_start = time.time()
                wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                data, points = wechat_detector.detectAndDecode(region)
                
                for j, d in enumerate(data):
                    if d:
                        # Adjust points coordinates to original image if available
                        adjusted_points = None
                        if j < len(points):
                            adjusted_points = points[j].copy()
                            adjusted_points[:, 0] += x  # Add x offset to all x coordinates
                            adjusted_points[:, 1] += y  # Add y offset to all y coordinates
                        
                        results.append({
                            "data": d,
                            "bounding_box": adjusted_points.tolist() if adjusted_points is not None else None,
                            "method": "wechat_qrcode",
                            "preprocessing": f"region_{i}",
                            "processing_time": time.time() - wechat_start
                        })
                        return results
            except (AttributeError, cv2.error):
                # WeChatQRCode not available in this OpenCV build
                pass
        
        return results
    
    def _try_pil_enhancements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Try PIL-based image enhancements"""
        results = []
        
        # Convert OpenCV image to PIL
        if len(image.shape) > 2 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Define enhancement methods
        enhancement_methods = [
            ("pil_contrast", lambda img: ImageEnhance.Contrast(img).enhance(2.0)),
            ("pil_brightness", lambda img: ImageEnhance.Brightness(img).enhance(1.5)),
            ("pil_sharpness", lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
            ("pil_edge_enhance", lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
            ("pil_detail", lambda img: img.filter(ImageFilter.DETAIL))
        ]
        
        # Try each enhancement method
        for name, enhance_func in enhancement_methods:
            # Apply enhancement
            enhanced_pil = enhance_func(pil_image)
            
            # Convert back to OpenCV format
            enhanced = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
            
            # Try OpenCV QRCodeDetector
            opencv_start = time.time()
            data, bbox, _ = self.qr_detector.detectAndDecode(enhanced)
            if data:
                results.append({
                    "data": data,
                    "bounding_box": bbox.tolist() if bbox is not None else None,
                    "method": "opencv",
                    "preprocessing": name,
                    "processing_time": time.time() - opencv_start
                })
                return results
            
            # Try WeChatQRCode detector if available
            try:
                wechat_start = time.time()
                wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                data, points = wechat_detector.detectAndDecode(enhanced)
                
                for i, d in enumerate(data):
                    if d:
                        bbox = points[i].tolist() if i < len(points) else None
                        results.append({
                            "data": d,
                            "bounding_box": bbox,
                            "method": "wechat_qrcode",
                            "preprocessing": name,
                            "processing_time": time.time() - wechat_start
                        })
                        return results
            except (AttributeError, cv2.error):
                # WeChatQRCode not available in this OpenCV build
                pass
        
        return results
    
    # Preprocessing methods
    def _preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) > 2 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _preprocess_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply binary threshold"""
        gray = self._preprocess_grayscale(image)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _preprocess_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive threshold"""
        gray = self._preprocess_grayscale(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    def _preprocess_enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using histogram equalization"""
        gray = self._preprocess_grayscale(image)
        return cv2.equalizeHist(gray)
    
    def _preprocess_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp mask"""
        if len(image.shape) > 2 and image.shape[2] == 3:
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
