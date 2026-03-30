"""
Brain Tumor Detection - File Path Diagnostic & Fix Script
Run this from your backend directory to diagnose the issue
"""

import os
import sys

def check_directory_structure():
    """Check if the required directories exist"""
    print("=" * 60)
    print("🔍 CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}\n")
    
    # Check for static folder
    static_path = os.path.join(current_dir, 'static')
    outputs_path = os.path.join(static_path, 'outputs')
    
    checks = {
        'static folder': os.path.exists(static_path),
        'static/outputs folder': os.path.exists(outputs_path),
    }
    
    for name, exists in checks.items():
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{status}: {name}")
    
    return static_path, outputs_path

def check_image_files(outputs_path):
    """Check if image files exist in outputs folder"""
    print("\n" + "=" * 60)
    print("🖼️  CHECKING IMAGE FILES")
    print("=" * 60)
    
    expected_images = ['clahe.png', 'dtcwt.png', 'loggabor.png', 
                       'entropy.png', 'roc.png', 'umap.png']
    
    if not os.path.exists(outputs_path):
        print(f"❌ Outputs folder doesn't exist: {outputs_path}")
        return False
    
    files_in_outputs = os.listdir(outputs_path)
    print(f"Files in outputs folder: {len(files_in_outputs)}")
    
    for img in expected_images:
        img_path = os.path.join(outputs_path, img)
        exists = os.path.exists(img_path)
        status = "✅" if exists else "❌"
        size = f"({os.path.getsize(img_path)} bytes)" if exists else ""
        print(f"{status} {img} {size}")
    
    if files_in_outputs:
        print(f"\nActual files found: {files_in_outputs}")
    else:
        print("\n⚠️  No files found in outputs folder!")
    
    return len(files_in_outputs) > 0

def check_flask_app():
    """Check Flask app configuration"""
    print("\n" + "=" * 60)
    print("🔧 CHECKING FLASK APP CONFIGURATION")
    print("=" * 60)
    
    app_py_path = 'app.py'
    
    if not os.path.exists(app_py_path):
        print(f"❌ app.py not found in current directory")
        print(f"   Make sure you're running this from the backend folder")
        return
    
    with open(app_py_path, 'r') as f:
        content = f.read()
    
    # Check for static_folder configuration
    if "static_folder='static'" in content or 'static_folder="static"' in content:
        print("✅ Static folder explicitly configured")
    elif "Flask(__name__)" in content:
        print("⚠️  Using default static folder configuration")
    else:
        print("❌ Could not determine static folder configuration")
    
    # Check for outputs directory creation
    if "os.makedirs" in content and "static/outputs" in content:
        print("✅ Code creates outputs directory")
    else:
        print("⚠️  No code to create outputs directory found")
    
    # Check image saving code
    if "plt.savefig" in content or "cv2.imwrite" in content:
        print("✅ Image saving code found")
    else:
        print("⚠️  No image saving code found")

def create_missing_directories():
    """Create missing directories"""
    print("\n" + "=" * 60)
    print("🛠️  CREATING MISSING DIRECTORIES")
    print("=" * 60)
    
    dirs_to_create = [
        'static',
        'static/outputs',
        'templates'
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"✅ Created: {dir_path}")
        else:
            print(f"✓  Already exists: {dir_path}")

def generate_test_images(outputs_path):
    """Generate test placeholder images"""
    print("\n" + "=" * 60)
    print("🎨 GENERATING TEST IMAGES")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        image_names = ['clahe', 'dtcwt', 'loggabor', 'entropy', 'roc', 'umap']
        
        for name in image_names:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'{name.upper()}\nTest Image', 
                   ha='center', va='center', fontsize=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            save_path = os.path.join(outputs_path, f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Created: {name}.png")
        
        print("\n✅ All test images created successfully!")
        return True
        
    except ImportError:
        print("❌ matplotlib not installed. Run: pip install matplotlib")
        return False

def show_fix_instructions():
    """Show instructions to fix the issue"""
    print("\n" + "=" * 60)
    print("📋 FIXING YOUR APP.PY")
    print("=" * 60)
    
    fix_code = """
# Add this at the top of your app.py after imports:
import os

# Create outputs directory if it doesn't exist
OUTPUTS_DIR = os.path.join('static', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# When saving images in your prediction function, use:
# Example:
def save_visualization(image, filename):
    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path)  # or cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")  # For debugging
    
# Make sure your Flask app is configured correctly:
app = Flask(__name__, static_folder='static')
"""
    
    print(fix_code)

def main():
    """Main diagnostic function"""
    print("\n🧠 BRAIN TUMOR DETECTION - IMAGE LOADING DIAGNOSTIC\n")
    
    # Check directory structure
    static_path, outputs_path = check_directory_structure()
    
    # Check image files
    images_exist = check_image_files(outputs_path)
    
    # Check Flask configuration
    check_flask_app()
    
    # Offer to create missing directories
    print("\n" + "=" * 60)
    response = input("Would you like to create missing directories? (y/n): ")
    if response.lower() == 'y':
        create_missing_directories()
    
    # Offer to generate test images
    if not images_exist:
        print("\n" + "=" * 60)
        response = input("Would you like to generate test placeholder images? (y/n): ")
        if response.lower() == 'y':
            generate_test_images(outputs_path)
    
    # Show fix instructions
    show_fix_instructions()
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Make sure the outputs directory exists")
    print("2. Update your app.py to save images to static/outputs/")
    print("3. Restart your Flask server")
    print("4. Hard refresh your browser (Ctrl+Shift+R)")
    print("\nIf images still don't show, check that your prediction")
    print("function actually runs and saves the images!")

if __name__ == "__main__":
    main()