#!/usr/bin/env python3
"""
InfoMask - Steganography Web Application Startup Script
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import PIL
        import numpy
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def main():
    print("=" * 50)
    print("InfoMask - Steganography Web Application")
    print("=" * 50)
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("\nPlease install dependencies manually:")
            print("pip install -r requirements.txt")
            return
    
    print("\nStarting the application...")
    print("The website will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the Flask app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main() 