from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import base64
import os
import time
from scipy.fft import dct, idct

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    valid_chars = 0
    total_chars = 0
    
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            try:
                char_code = int(byte, 2)
                total_chars += 1
                # Only add printable characters
                if 32 <= char_code <= 126:
                    text += chr(char_code)
                    valid_chars += 1
                else:
                    # If we encounter non-printable characters, this might not be valid text
                    break
            except ValueError:
                continue
    
    # Check if we have a reasonable amount of valid characters
    if total_chars > 0 and valid_chars / total_chars < 0.8:
        return ""
    
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
    
    # Additional validation: check if the text looks like meaningful content
    if text:
        # Check if text contains mostly alphanumeric characters and common punctuation
        valid_chars = sum(1 for c in text if c.isalnum() or c in ' .,!?-_\'\"')
        if len(text) > 0 and valid_chars / len(text) < 0.7:
            return ""
        
        # Check if text is too short (likely noise)
        if len(text) < 2:
            return ""
    
    return text

def hide_text_in_image_dct(image, text):
    """Hide text in image using DCT-based steganography"""
    # Convert text to binary
    binary_text = text_to_binary(text)
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for DCT processing
    if len(img_array.shape) == 3:
        # Convert RGB to grayscale using standard weights
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img_array
    
    # Ensure image dimensions are multiples of 8 for DCT blocks
    h, w = gray.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    
    if h_pad > 0 or w_pad > 0:
        gray = np.pad(gray, ((0, h_pad), (0, w_pad)), mode='edge')
    
    # Process in 8x8 blocks
    stego_img = gray.copy()
    bit_index = 0
    
    for i in range(0, gray.shape[0] - 7, 8):
        for j in range(0, gray.shape[1] - 7, 8):
            if bit_index >= len(binary_text):
                break
            
            # Extract 8x8 block
            block = gray[i:i+8, j:j+8].astype(np.float32)
            
            # Apply DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Embed bit in DCT coefficients (using middle frequency coefficients)
            if bit_index < len(binary_text):
                bit = int(binary_text[bit_index])
                # Use coefficient at position (3,4) for embedding
                if bit == 1:
                    dct_block[3, 4] = abs(dct_block[3, 4]) + 10
                else:
                    dct_block[3, 4] = -abs(dct_block[3, 4]) - 10
                bit_index += 1
            
            # Apply inverse DCT
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            
            # Place back in image
            stego_img[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)
        
        if bit_index >= len(binary_text):
            break
    
    # Remove padding
    if h_pad > 0 or w_pad > 0:
        stego_img = stego_img[:h, :w]
    
    # Convert back to RGB for PIL
    if len(img_array.shape) == 3:
        # Convert grayscale back to 3-channel RGB
        stego_img = np.stack([stego_img, stego_img, stego_img], axis=2)
    
    stego_img = stego_img.astype(np.uint8)
    return Image.fromarray(stego_img)

def extract_text_from_image_dct(image):
    """Extract hidden text from DCT-based steganography"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for DCT processing
    if len(img_array.shape) == 3:
        # Convert RGB to grayscale using standard weights
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img_array
    
    # Ensure image dimensions are multiples of 8 for DCT blocks
    h, w = gray.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    
    if h_pad > 0 or w_pad > 0:
        gray = np.pad(gray, ((0, h_pad), (0, w_pad)), mode='edge')
    
    # Process in 8x8 blocks
    binary_text = ""
    
    for i in range(0, gray.shape[0] - 7, 8):
        for j in range(0, gray.shape[1] - 7, 8):
            # Extract 8x8 block
            block = gray[i:i+8, j:j+8].astype(np.float32)
            
            # Apply DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Extract bit from DCT coefficient at position (3,4)
            coeff = dct_block[3, 4]
            if coeff > 0:
                binary_text += '1'
            else:
                binary_text += '0'
            
            # Check for null terminator (8 consecutive zeros)
            if len(binary_text) >= 8 and binary_text[-8:] == '00000000':
                # Remove the null terminator
                binary_text = binary_text[:-8]
                break
        
        # Check for null terminator after each row
        if len(binary_text) >= 8 and binary_text[-8:] == '00000000':
            break
    
    # Convert binary back to text
    text = binary_to_text(binary_text)
    
    # Additional validation: check if the text looks like meaningful content
    if text:
        # Check if text contains mostly alphanumeric characters and common punctuation
        valid_chars = sum(1 for c in text if c.isalnum() or c in ' .,!?-_\'\"')
        if len(text) > 0 and valid_chars / len(text) < 0.7:
            return ""
        
        # Check if text is too short (likely noise)
        if len(text) < 2:
            return ""
    
    return text

def image_to_binary(image):
    """Convert image to binary string using LSB - OPTIMIZED"""
    img_array = np.array(image)
    
    # Convert to RGB if necessary
    if len(img_array.shape) == 3:
        # Flatten the image array
        flat_img = img_array.flatten()
    else:
        # Convert grayscale to RGB
        flat_img = np.stack([img_array, img_array, img_array], axis=2).flatten()
    
    # Vectorized binary conversion - much faster
    # Convert to uint8 to ensure proper range
    flat_img = flat_img.astype(np.uint8)
    
    # OPTIMIZED: Direct numpy unpackbits - fastest method
    binary_array = np.unpackbits(flat_img)
    
    # Convert to string efficiently
    binary_data = ''.join(binary_array.astype(str))
    
    return binary_data

def binary_to_image(binary_data, original_shape):
    """Convert binary string back to image - OPTIMIZED"""
    # Calculate number of pixels needed
    total_pixels = original_shape[0] * original_shape[1] * original_shape[2]
    
    # Ensure binary data length is multiple of 8
    if len(binary_data) % 8 != 0:
        binary_data = binary_data.ljust((len(binary_data) + 7) // 8 * 8, '0')
    
    # Convert binary string to numpy array of bits
    binary_array = np.array(list(binary_data), dtype=np.uint8)
    
    # Reshape to groups of 8 bits
    binary_8bit = binary_array[:total_pixels * 8].reshape(-1, 8)
    
    # Convert each 8-bit group to decimal using vectorized operations
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    pixels = np.sum(binary_8bit * powers, axis=1, dtype=np.uint8)
    
    # Pad with zeros if necessary
    if len(pixels) < total_pixels:
        padding = np.zeros(total_pixels - len(pixels), dtype=np.uint8)
        pixels = np.concatenate([pixels, padding])
    
    # Reshape to original image dimensions
    img_array = pixels[:total_pixels].reshape(original_shape)
    
    return Image.fromarray(img_array.astype(np.uint8))

def hide_image_in_image(cover_image, secret_image):
    """Hide one image inside another using LSB steganography - OPTIMIZED"""
    # Convert both images to RGB
    cover_array = np.array(cover_image.convert('RGB'))
    secret_array = np.array(secret_image.convert('RGB'))
    
    # Store original secret image dimensions
    original_secret_shape = secret_array.shape
    
    # Resize secret image to fit in cover image (but keep it reasonable)
    cover_height, cover_width = cover_array.shape[:2]
    
    # Calculate maximum secret image size that can fit
    max_secret_pixels = (cover_array.size - 40) // 8  # 40 bits for header
    max_secret_size = int(np.sqrt(max_secret_pixels // 3))  # 3 channels
    
    # Resize secret image to fit within limits - OPTIMIZED
    if secret_array.size > max_secret_pixels:
        # Scale down the secret image using faster resampling
        scale_factor = np.sqrt(max_secret_pixels / secret_array.size)
        new_width = max(1, int(secret_array.shape[1] * scale_factor))
        new_height = max(1, int(secret_array.shape[0] * scale_factor))
        # Use NEAREST for faster resizing (good enough for steganography)
        secret_resized = secret_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        secret_array = np.array(secret_resized)
    
    # Convert secret image to binary
    secret_binary = image_to_binary(Image.fromarray(secret_array))
    
    # Add header information (width, height, channels)
    header = f"{secret_array.shape[0]:016b}{secret_array.shape[1]:016b}{secret_array.shape[2]:08b}"
    full_binary = header + secret_binary
    
    # Flatten cover image
    flat_cover = cover_array.flatten()
    
    # Check if cover image can hold the data
    if len(full_binary) > len(flat_cover):
        raise ValueError("Cover image is too small to hide the secret image")
    
    # OPTIMIZED: Vectorized LSB hiding - much faster
    # Convert binary string to numpy array
    binary_array = np.array(list(full_binary), dtype=np.uint8)
    
    # Ensure we don't exceed cover image size
    binary_array = binary_array[:len(flat_cover)]
    
    # Vectorized LSB operation: clear LSB and set new bit
    flat_cover[:len(binary_array)] = (flat_cover[:len(binary_array)] & 0xFE) | binary_array
    
    # Reshape back to original dimensions
    stego_array = flat_cover.reshape(cover_array.shape)
    
    return Image.fromarray(stego_array.astype(np.uint8))

def extract_image_from_image(stego_image):
    """Extract hidden image from stego image - OPTIMIZED"""
    stego_array = np.array(stego_image.convert('RGB'))
    flat_stego = stego_array.flatten()
    
    # OPTIMIZED: Vectorized header extraction
    if len(flat_stego) < 40:
        raise ValueError("Invalid stego image - header too short")
    
    # Extract header bits using vectorized operations
    header_bits = flat_stego[:40] & 1
    
    # Parse header
    height = int(''.join(header_bits[:16].astype(str)), 2)
    width = int(''.join(header_bits[16:32].astype(str)), 2)
    channels = int(''.join(header_bits[32:40].astype(str)), 2)
    
    # Calculate total bits needed for the secret image
    total_bits = height * width * channels * 8
    
    # Check if we have enough data
    if 40 + total_bits > len(flat_stego):
        raise ValueError("Invalid stego image - data too short")
    
    # OPTIMIZED: Vectorized secret data extraction
    secret_bits = flat_stego[40:40 + total_bits] & 1
    secret_binary = ''.join(secret_bits.astype(str))
    
    # Convert binary back to image
    secret_shape = (height, width, channels)
    secret_image = binary_to_image(secret_binary, secret_shape)
    
    return secret_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hide', methods=['POST'])
def hide_text():
    try:
        # Get uploaded image, text, and encryption method
        image_file = request.files['image']
        text = request.form['text']
        encryption_method = request.form.get('encryption_method', 'lsb')  # Default to LSB
        
        if not image_file or not text:
            return jsonify({'error': 'Please provide both image and text'}), 400
        
        # Open and process image
        image = Image.open(image_file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Hide text in image using selected method
        if encryption_method == 'dct':
            stego_image = hide_text_in_image_dct(image, text)
        else:  # Default to LSB
            stego_image = hide_text_in_image(image, text)
        
        # Save to bytes
        img_io = io.BytesIO()
        stego_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Convert to base64 for frontend
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        # Save to disk as backup
        timestamp = int(time.time())
        filename = f"steganographed_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        stego_image.save(filepath, 'PNG')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'message': 'Text hidden successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/extract', methods=['POST'])
def extract_text():
    try:
        # Get uploaded image and encryption method
        image_file = request.files['image']
        encryption_method = request.form.get('encryption_method', 'lsb')  # Default to LSB
        
        if not image_file:
            return jsonify({'error': 'Please provide an image'}), 400
        
        # Open and process image
        image = Image.open(image_file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text from image using selected method
        if encryption_method == 'dct':
            hidden_text = extract_text_from_image_dct(image)
        else:  # Default to LSB
            hidden_text = extract_text_from_image(image)
        
        print(f"Extracted text: '{hidden_text}' (length: {len(hidden_text)})")
        
        if not hidden_text:
            return jsonify({'error': 'No hidden text found in this image. This image either contains no hidden text or the text was not hidden using this tool.'}), 400
        
        return jsonify({
            'success': True,
            'text': hidden_text,
            'message': 'Text extracted successfully!'
        })
        
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return jsonify({'error': f'Error extracting text: {str(e)}'}), 400

@app.route('/hide-image', methods=['POST'])
def hide_image():
    """Hide one image inside another"""
    try:
        # Get uploaded images
        cover_file = request.files['cover_image']
        secret_file = request.files['secret_image']
        
        if not cover_file or not secret_file:
            return jsonify({'error': 'Please provide both cover and secret images'}), 400
        
        # Open and process images
        cover_image = Image.open(cover_file.stream)
        secret_image = Image.open(secret_file.stream)
        
        # Convert to RGB if necessary
        if cover_image.mode != 'RGB':
            cover_image = cover_image.convert('RGB')
        if secret_image.mode != 'RGB':
            secret_image = secret_image.convert('RGB')
        
        # Hide secret image in cover image
        stego_image = hide_image_in_image(cover_image, secret_image)
        
        # Save to bytes
        img_io = io.BytesIO()
        stego_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Convert to base64 for frontend
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        # Save to disk as backup
        timestamp = int(time.time())
        filename = f"image_steganographed_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        stego_image.save(filepath, 'PNG')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'message': 'Image hidden successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/extract-image', methods=['POST'])
def extract_image():
    """Extract hidden image from stego image"""
    try:
        # Get uploaded stego image
        stego_file = request.files['stego_image']
        
        if not stego_file:
            return jsonify({'error': 'Please provide a stego image'}), 400
        
        # Open and process image
        stego_image = Image.open(stego_file.stream)
        
        # Convert to RGB if necessary
        if stego_image.mode != 'RGB':
            stego_image = stego_image.convert('RGB')
        
        # Extract hidden image
        secret_image = extract_image_from_image(stego_image)
        
        # Save to bytes
        img_io = io.BytesIO()
        secret_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Convert to base64 for frontend
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        # Save to disk as backup
        timestamp = int(time.time())
        filename = f"extracted_image_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        secret_image.save(filepath, 'PNG')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'message': 'Image extracted successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error extracting image: {str(e)}'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the uploads directory"""
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 