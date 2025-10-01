#!/usr/bin/env python3
"""
Test script for steganography functionality
"""

from PIL import Image, ImageDraw
import numpy as np
import io

def text_to_binary(text):
    """Convert text to binary string"""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary + '00000000'  # Add null terminator

def binary_to_text(binary):
    """Convert binary string back to text"""
    # Ensure binary string length is a multiple of 8
    if len(binary) % 8 != 0:
        # Pad with zeros if necessary
        binary = binary.ljust((len(binary) + 7) // 8 * 8, '0')
    
    text = ""
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            try:
                char_code = int(byte, 2)
                # Only add printable characters
                if 32 <= char_code <= 126:
                    text += chr(char_code)
            except ValueError:
                continue
    
    return text

def hide_text_in_image(image, text):
    """Hide text in image using LSB steganography"""
    # Convert text to binary
    binary_text = text_to_binary(text)
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Check if image can hold the text
    if len(binary_text) > img_array.size:
        raise ValueError("Text too long for this image")
    
    # Flatten the image array
    flat_img = img_array.flatten()
    
    # Hide the binary text in the least significant bits
    for i, bit in enumerate(binary_text):
        flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
    
    # Reshape back to original dimensions
    stego_img = flat_img.reshape(img_array.shape)
    
    return Image.fromarray(stego_img.astype(np.uint8))

def extract_text_from_image(image):
    """Extract hidden text from image"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Flatten the image array
    flat_img = img_array.flatten()
    
    # Extract binary text from least significant bits
    binary_text = ""
    for i in range(len(flat_img)):
        bit = flat_img[i] & 1
        binary_text += str(bit)
        
        # Check for null terminator (8 consecutive zeros)
        if len(binary_text) >= 8 and binary_text[-8:] == '00000000':
            # Remove the null terminator
            binary_text = binary_text[:-8]
            break
    
    # Convert binary back to text
    text = binary_to_text(binary_text)
    return text

def test_steganography():
    """Test the steganography algorithm"""
    print("Testing steganography algorithm...")
    
    # Create a test image
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test", fill='black')
    
    # Test text
    test_text = "Hello, this is a secret message!"
    print(f"Original text: {test_text}")
    
    # Hide text
    stego_img = hide_text_in_image(img, test_text)
    print("Text hidden successfully")
    
    # Extract text
    extracted_text = extract_text_from_image(stego_img)
    print(f"Extracted text: {extracted_text}")
    
    # Verify
    if test_text == extracted_text:
        print("✅ SUCCESS: Text extraction works correctly!")
        return True
    else:
        print("❌ FAILED: Text extraction failed!")
        print(f"Expected: '{test_text}'")
        print(f"Got: '{extracted_text}'")
        return False

if __name__ == "__main__":
    test_steganography() 