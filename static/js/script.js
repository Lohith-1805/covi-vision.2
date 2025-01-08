document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const submitButton = document.getElementById('submitButton');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const previewImage = document.getElementById('previewImage');
    const resultContainer = document.getElementById('resultContainer');
    const predictionResult = document.getElementById('predictionResult');

    // Handle click on drop zone
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag and drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // Handle file input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    // Handle file processing
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                imagePreviewContainer.style.display = 'block';
                submitButton.disabled = false;
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload an image file');
        }
    }

    // Handle form submission
    submitButton.addEventListener('click', async () => {
        const file = fileInput.files[0] || new File([], '');
        const formData = new FormData();
        formData.append('image', file);

        try {
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log('Response:', data);

            if (data.success) {
                predictionResult.textContent = data.prediction;
                resultContainer.style.display = 'block';
            } else {
                alert('Error: ' + (data.error || 'Unable to process image'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image. Please try again.');
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-microscope"></i> Analyze Image';
        }
    });
});