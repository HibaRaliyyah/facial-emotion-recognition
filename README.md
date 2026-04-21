# Facial Emotion Recognition

## Project Overview
This project aims to provide a comprehensive facial emotion recognition system using modern machine learning techniques. The objective is to develop a model that can accurately identify and classify facial expressions based on images.

## Installation
To set up the Facial Emotion Recognition project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/HibaRaliyyah/facial-emotion-recognition.git
   ```
2. Navigate into the project directory:
   ```bash
   cd facial-emotion-recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Endpoints
The following API endpoints are available for the Facial Emotion Recognition service:

1. **POST /api/recognize**
   - Description: Sends an image to the server for emotion recognition.
   - Request Body:
     ```json
     {
       "image": "base64_encoded_image"
     }
     ```
   - Response:
     ```json
     {
       "emotion": "happy",
       "confidence": 0.95
     }
     ```

## Usage Examples
Here’s how to use the API:

### Recognizing Emotion from an Image
```python
import requests

# Replace with the actual endpoint
url = 'http://localhost:5000/api/recognize'

# Load your image and encode it in base64
with open('path_to_your_image.jpg', 'rb') as image:
    encoded_image = base64.b64encode(image.read()).decode('utf-8')

response = requests.post(url, json={'image': encoded_image})
print(response.json())
```

Make sure to replace `path_to_your_image.jpg` with the actual path to your image file.

## Conclusion
This project serves as a foundation for understanding facial emotion recognition techniques and can be extended for various applications, including but not limited to, enhancing user experience in technology and improving interactions in social robotics.