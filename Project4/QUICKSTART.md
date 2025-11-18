# Quick Start Guide - MonReader Page Flip Detection

## Fast Setup (15 minutes)

### 1. Install Dependencies

#### Option A: With CUDA GPU Support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tensorflow keras scikit-learn Pillow pandas numpy matplotlib seaborn
```

#### Option B: CPU Only
```bash
pip install -r requirements_monreader.txt
```

### 2. Verify GPU (Optional)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 3. Prepare Your Data

**Directory Structure:**
```
project/
├── images/
│   ├── training/
│   │   ├── flip/          # Put flip images here
│   │   └── notflip/       # Put non-flip images here
│   └── testing/
│       ├── flip/
│       └── notflip/
└── images_new/
    ├── Flipping/          # External flip images (optional)
    └── NotFlipping/       # External non-flip images (optional)
```

### 4. Run the Complete Pipeline

**Option 1: Quick Training (CNN Only)**
```bash
# Step 1: Prepare data
jupyter notebook monReader_exploration.ipynb
# Run all cells to generate pickle files

# Step 2: Train CNN model
jupyter notebook simpleCNN_augmented_images.ipynb
# Run all cells (~30-45 min on GPU)
```

**Option 2: Full Pipeline (CNN + Sequence Detection)**
```bash
# Step 1: Data preparation
jupyter notebook monReader_exploration.ipynb

# Step 2: Single frame CNN
jupyter notebook simpleCNN_augmented_images.ipynb

# Step 3: Sequence detection
jupyter notebook sequence_flipping.ipynb
```

## Expected Results

After running the pipeline, you should see:
- ✅ **F1 Score: >99%** for single frame detection
- ✅ **High Accuracy: >95%** on validation set
- ✅ **Feature maps** showing learned filters
- ✅ **Confusion matrix** with minimal errors
- ✅ **Trained model** saved as `.pth` file
- ✅ **Sequence detection** working for video frames

## Notebook Execution Order

```
1. monReader_exploration.ipynb (REQUIRED)
   ├─> Loads images from directories
   ├─> Performs EDA
   └─> Generates: resized_combined_dataset.pkl
   
2. simpleCNN_augmented_images.ipynb (REQUIRED)
   ├─> Trains CNN on flip detection
   ├─> Applies data augmentation
   └─> Achieves: F1 > 99%
   
3. sequence_flipping.ipynb (OPTIONAL)
   ├─> Creates temporal sequences
   ├─> Trains CNN+LSTM
   └─> Enables: Video frame analysis
```

## Key Files

| File | Purpose | Size | When to Use |
|------|---------|------|-------------|
| `monReader_exploration.ipynb` | Data prep & EDA | 3.9M | Always - first step |
| `simpleCNN_augmented_images.ipynb` | CNN training | 456K | Single frame detection |
| `sequence_flipping.ipynb` | Sequence modeling | 248K | Video/stream processing |

## Quick Test After Training

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('flip_detector_model.pth'))
model.eval()

# Test transform
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Test on new image
test_image = Image.open('path/to/test_image.jpg')
test_tensor = test_transform(test_image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(test_tensor)
    _, prediction = torch.max(output, 1)
    confidence = torch.softmax(output, dim=1)[0][prediction].item()
    
    result = "FLIP" if prediction.item() == 1 else "NOT FLIP"
    print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

## Quick Customization

### Change Image Size
```python
# In both CNN and sequence notebooks
augmented_transform = transforms.Compose([
    transforms.Resize((512, 512)),      # Larger images
    transforms.RandomCrop(480),         # Adjust crop size
    # ... other transforms
])
```

### Adjust Training Epochs
```python
# In simpleCNN_augmented_images.ipynb
num_epochs = 100  # Increase from 60
```

### Modify Batch Size
```python
# Reduce if memory issues
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Increase for faster training (if GPU allows)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Change Sequence Length
```python
# In sequence_flipping.ipynb
sequences, seq_labels = create_sequence_dataset(
    images, 
    labels, 
    sequence_length=5  # Try 4, 5, or 6
)
```

## Troubleshooting

**Problem**: CUDA out of memory
```python
# Solution 1: Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Solution 2: Clear cache between runs
import gc
torch.cuda.empty_cache()
gc.collect()

# Solution 3: Use CPU
device = torch.device("cpu")
```

**Problem**: Low accuracy
- Add more training data (check flip vs notflip balance)
- Increase epochs (60 → 100)
- Verify image quality and labels
- Check data augmentation settings

**Problem**: Images not loading
```bash
# Verify directory structure
ls images/training/flip/
ls images/training/notflip/

# Check file permissions
chmod -R 755 images/
```

**Problem**: Pickle file errors
```python
# Regenerate pickle files
# Run monReader_exploration.ipynb from start
```

**Problem**: Model not improving
```python
# Increase learning rate
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Or decrease if overshooting
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

## Understanding Model Output

### SimpleCNN Output
```
Model output shape: (batch_size, 2)
Class 0: Not Flip
Class 1: Flip

Example:
tensor([[0.1234, 0.8766]])  # 87.66% confidence it's a flip
```

### Confusion Matrix Interpretation
```
                Predicted
              Not Flip  Flip
Actual NotFlip   TN      FP
       Flip      FN      TP

Good model: High TN and TP, Low FP and FN
```

### Feature Maps
- **Conv1**: Detects edges and basic patterns
- **Conv2**: Detects complex shapes and motion blur
- **Rainbow colormap**: Shows activation intensity

## Performance Benchmarks

| Hardware | Training Time (60 epochs) | Inference Time |
|----------|--------------------------|----------------|
| NVIDIA RTX 3080 | ~30 min | <5ms per frame |
| NVIDIA GTX 1660 | ~45 min | <10ms per frame |
| CPU (Intel i7) | ~6 hours | ~50ms per frame |

## Data Requirements

### Minimum Dataset Size
- **Training**: 200 images per class (flip + notflip)
- **Validation**: 50 images per class
- **Testing**: 30 images per class

### Recommended Dataset Size
- **Training**: 500+ images per class
- **Validation**: 100+ images per class
- **Testing**: 50+ images per class

### Image Quality Guidelines
- ✅ Clear, well-lit images
- ✅ Consistent resolution (min 200×200)
- ✅ Varied angles and positions
- ✅ Different book types and colors
- ❌ Avoid extremely blurry images
- ❌ Avoid completely dark/bright images

## Next Steps

1. 📊 **Collect more data** if accuracy is low
2. 🔄 **Fine-tune hyperparameters** (learning rate, batch size)
3. 🎯 **Test on real videos** using sequence detector
4. 📱 **Deploy to mobile** using PyTorch Mobile or TensorFlow Lite
5. 🚀 **Create REST API** for cloud deployment
6. 📈 **Monitor performance** on production data

## Production Deployment

### Export Model for Mobile
```python
# PyTorch Mobile
model_scripted = torch.jit.script(model)
model_scripted.save('flip_detector_mobile.pt')

# ONNX format
torch.onnx.export(model, dummy_input, 'flip_detector.onnx')
```

### REST API Example (Flask)
```python
from flask import Flask, request, jsonify
import torch
from PIL import Image

app = Flask(__name__)
model = SimpleCNN()
model.load_state_dict(torch.load('flip_detector_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['image'])
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.max(output, 1)[1].item()
    
    return jsonify({'flip': bool(prediction)})

if __name__ == '__main__':
    app.run(port=5000)
```

## Need Help?

- Check the main README_MONREADER.md for detailed documentation
- Review feature maps to debug model learning
- Test with smaller dataset first
- Contact: schakr18@umd.edu

## Success Criteria

✅ **F1 Score**: >99% on validation set  
✅ **Training Loss**: Decreasing steadily  
✅ **Validation Loss**: Not increasing (no overfitting)  
✅ **Feature Maps**: Show clear patterns  
✅ **Inference Speed**: Real-time capable (<50ms)  
✅ **Robustness**: Works on varied test images  

---

**Pro Tip**: Start with the default settings and only adjust hyperparameters if results are unsatisfactory. The default configuration achieves >99% F1 score!
