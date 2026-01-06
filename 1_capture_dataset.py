"""
MODULE 1: CAPTURE DATASET - FIXED
Purpose: Capture face images WITHOUT green box in saved files

Bug fixed: Extract face BEFORE drawing rectangle
"""

import cv2
import os
import time

print("\n" + "="*70)
print("MODULE 1: DATASET CAPTURE")
print("="*70)

# Get person's name
person_name = input("\nEnter person's name: ").strip()

if not person_name:
    print("❌ Name cannot be empty!")
    exit()

# Create folders
dataset_folder = "dataset"
person_folder = os.path.join(dataset_folder, person_name)

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

if os.path.exists(person_folder):
    print(f"\n⚠️  '{person_name}' already exists!")
    overwrite = input("Overwrite? (yes/no): ").lower()
    if overwrite != 'yes':
        print("Cancelled")
        exit()
    # Clear old images
    for old_file in os.listdir(person_folder):
        os.remove(os.path.join(person_folder, old_file))
else:
    os.makedirs(person_folder)

# Setup webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("📹 System will capture 100 images")
print("📹 Move your head slowly:")
print("   - Look straight")
print("   - Turn slightly left/right")
print("   - Move slightly up/down")
print()
print("Starting in 3 seconds...")
print("="*70)
time.sleep(3)

# Capture settings
captured_count = 0
target_images = 100
frame_counter = 0
capture_every = 3

print("\n🎥 Capturing...")

while captured_count < target_images:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error reading frame")
        break
    
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process face
    face_detected = False
    for (x, y, w, h) in faces:
        face_detected = True
        
        # ============================================
        # CRITICAL FIX: Extract face BEFORE drawing
        # ============================================
        frame_counter += 1
        if frame_counter % capture_every == 0:
            # Extract face with padding
            padding = 20
            y1 = max(0, y - padding)
            y2 = min(frame_h, y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame_w, x + w + padding)
            
            # Extract from ORIGINAL frame (no rectangles drawn yet)
            face_img = frame[y1:y2, x1:x2].copy()
            
            # Resize to 128x128
            face_img_resized = cv2.resize(face_img, (128, 128))
            
            # Save clean image (no green box!)
            img_path = os.path.join(person_folder, f'{captured_count:03d}.jpg')
            cv2.imwrite(img_path, face_img_resized)
            
            captured_count += 1
            
            if captured_count % 10 == 0:
                print(f"  ✓ {captured_count}/{target_images} images captured")
        
        # NOW draw rectangle (only for display, not saved)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        break  # Only process first face
    
    # Display progress
    progress_pct = int((captured_count / target_images) * 100)
    progress_text = f"Progress: {captured_count}/{target_images} ({progress_pct}%)"
    
    cv2.putText(frame, progress_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if face_detected:
        cv2.putText(frame, "✓ Face Detected", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "✗ No Face", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press 'q' to quit", (30, frame_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Dataset Capture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n⚠️  Cancelled")
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print("\n" + "="*70)
print("CAPTURE SUMMARY")
print("="*70)

if captured_count >= 80:
    print("✅ SUCCESS!")
    print(f"   Name: {person_name}")
    print(f"   Images: {captured_count}")
    print(f"   Location: {person_folder}")
    print()
    print("📸 Images saved WITHOUT green box")
    print()
    print("NEXT STEPS:")
    print("1. Capture more people (run this script again)")
    print("2. When done, create zip file:")
    print(f"   zip -r dataset.zip {dataset_folder}")
    print("3. Upload dataset.zip to Google Colab")
else:
    print("⚠️  INCOMPLETE")
    print(f"   Only {captured_count} images")
    print("   Minimum 80 required")

print("="*70 + "\n")