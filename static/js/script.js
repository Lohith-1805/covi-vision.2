document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const resultDiv = document.getElementById('result');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Preview image before upload
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const file = imageInput.files[0];
        
        if (!file) {
            alert('Please select an image first!');
            return;
        }

        formData.append('image', file);
        
        try {
            // Show loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'block';
            }
            resultDiv.innerHTML = '';

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                resultDiv.innerHTML = `
                    <div class="alert alert-success">
                        <h4>Prediction Result:</h4>
                        <p>Diagnosis: ${data.prediction}</p>
                        <p>Confidence: ${data.confidence}</p>
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <p>Error: ${data.error}</p>
                    </div>
                `;
            }
        } catch (error) {
            resultDiv.innerHTML = `
                <div class="alert alert-danger">
                    <p>Error: Unable to process the image. Please try again.</p>
                </div>
            `;
        } finally {
            // Hide loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
        }
    });
});