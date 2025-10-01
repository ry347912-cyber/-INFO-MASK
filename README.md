InfoMask - Steganography Web Application
A modern web application that allows users to hide secret text messages in images using LSB (Least Significant Bit) steganography and extract hidden messages from images.

Features
Hide Text in Images: Upload any image and embed secret text messages using advanced steganography techniques
Extract Hidden Text: Upload images containing hidden messages to reveal the secret text
Modern UI: Beautiful, responsive design with drag-and-drop functionality
Multiple Format Support: Works with JPG, PNG, GIF, and BMP image formats
Real-time Processing: Fast and efficient text hiding/extraction with visual feedback
Download Results: Download processed images with hidden text
How Steganography Works
This application uses LSB (Least Significant Bit) steganography, which works by:

Converting the secret text into binary format
Modifying the least significant bits of each pixel's RGB values
The changes are imperceptible to the human eye
The hidden text can be extracted by reading the modified bits
Installation
Prerequisites
Python 3.7 or higher
pip (Python package installer)
Setup Instructions
Clone or download the project files

Install Python dependencies:

pip install -r requirements.txt
Run the application:

python app.py
Open your web browser and navigate to:

http://localhost:5000
Usage
Hiding Text in Images
Click on the "Hide Text" tab
Upload an image by clicking the upload area or dragging and dropping
Enter your secret message in the text area
Click "Hide Text in Image"
Download the processed image with hidden text
Extracting Hidden Text
Click on the "Extract Text" tab
Upload an image that contains hidden text
Click "Extract Hidden Text"
View the revealed secret message
Technical Details
Backend (Python Flask)
Flask: Web framework for handling HTTP requests
Pillow (PIL): Image processing library
NumPy: Numerical computing for efficient array operations
Werkzeug: WSGI utilities for file handling
Frontend (HTML/CSS/JavaScript)
Responsive Design: Works on desktop and mobile devices
Modern UI: Clean, intuitive interface with smooth animations
Drag & Drop: Easy file upload functionality
Real-time Feedback: Loading indicators and success/error messages
Steganography Algorithm
Converts text to binary using UTF-8 encoding
Modifies least significant bits of RGB pixel values
Uses null terminator (8 consecutive zeros) to mark text end
Supports images with sufficient pixel capacity
Security Features
Invisible Changes: Text modifications are imperceptible to human vision
Format Preservation: Maintains original image quality and format
Error Handling: Robust error handling for invalid inputs
File Size Limits: Prevents abuse with reasonable file size restrictions
File Structure
steganography-website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
└── uploads/              # Temporary file storage (auto-created)
Browser Compatibility
Chrome 60+
Firefox 55+
Safari 12+
Edge 79+
Troubleshooting
Common Issues
"Text too long for this image"

Use a larger image or shorter text message
The image needs enough pixels to store your text
"No hidden text found"

Ensure the image actually contains hidden text
Check that the image wasn't compressed or modified after steganography
Port already in use

Change the port in app.py line: app.run(debug=True, host='0.0.0.0', port=5001)
Installation errors

Ensure you have Python 3.7+ installed
Try upgrading pip: python -m pip install --upgrade pip
Limitations
Text length is limited by image pixel count
Works best with lossless image formats (PNG recommended)
JPEG compression may destroy hidden data
Very large images may take longer to process
Contributing
Feel free to submit issues, feature requests, or pull requests to improve this application.

License
This project is open source and available under the MIT License.

Note: This tool is for educational and legitimate purposes only. Always respect privacy and copyright laws when using steganography techniques.
