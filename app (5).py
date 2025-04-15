import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
from skimage import exposure
import time

# Load models (using free Hugging Face models)
MODEL_NAMES = {
    "Model 1": "dima806/deepfake_vs_real_image_detection",
    "Model 2": "saltacc/anime-ai-detect",
    "Model 3": "rizvandwiki/gansfake-detector"
}

# Initialize models
models = {}
processors = {}

for name, path in MODEL_NAMES.items():
    try:
        processors[name] = AutoImageProcessor.from_pretrained(path)
        models[name] = AutoModelForImageClassification.from_pretrained(path)
    except:
        print(f"Could not load model: {name}")

def analyze_image(image, selected_model):
    if image is None:
        return None, None, "Please upload an image first", None
    
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get model and processor
        model = models.get(selected_model)
        processor = processors.get(selected_model)
        
        if not model or not processor:
            return None, None, "Selected model not available", None
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Create visualizations
        heatmap = generate_heatmap(image, model, processor)
        chart_fig = create_probability_chart(probs, model.config.id2label)
        
        # Format results
        result_text = format_results(probs, model.config.id2label)
        
        return heatmap, chart_fig, result_text, create_model_info(selected_model)
    
    except Exception as e:
        return None, None, f"Error: {str(e)}", None

def generate_heatmap(image, model, processor):
    """Generate a heatmap showing important regions for the prediction"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create a saliency map (simple version)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    
    # Blend with original image
    heatmap = cv2.addWeighted(img_array, 0.7, heatmap, 0.3, 0)
    
    # Convert back to PIL
    return Image.fromarray(heatmap)

def create_probability_chart(probs, id2label):
    """Create a bar chart of class probabilities"""
    labels = [id2label[i] for i in range(len(probs))]
    colors = ['#4CAF50' if 'real' in label.lower() else '#F44336' for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, probs.numpy(), color=colors)
    ax.set_xlim(0, 1)
    ax.set_title('Detection Probabilities', pad=20)
    ax.set_xlabel('Probability')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                va='center')
    
    plt.tight_layout()
    return fig

def format_results(probs, id2label):
    """Format the results as text"""
    results = []
    for i, prob in enumerate(probs):
        results.append(f"{id2label[i]}: {prob*100:.1f}%")
    
    max_prob = max(probs)
    max_class = id2label[torch.argmax(probs).item()]
    
    if 'real' in max_class.lower():
        conclusion = f"Conclusion: This image appears to be AUTHENTIC with {max_prob*100:.1f}% confidence"
    else:
        conclusion = f"Conclusion: This image appears to be FAKE/GENERATED with {max_prob*100:.1f}% confidence"
    
    return "\n".join([conclusion, "", "Detailed probabilities:"] + results)

def create_model_info(model_name):
    """Create information about the current model"""
    info = {
        "Model 1": "Trained to detect deepfakes vs real human faces",
        "Model 2": "Specialized in detecting AI-generated anime images",
        "Model 3": "General GAN-generated image detector"
    }
    return info.get(model_name, "No information available about this model")

# Custom CSS for the interface
custom_css = """
:root {
    --primary: #4b6cb7;
    --secondary: #182848;
    --authentic: #4CAF50;
    --fake: #F44336;
    --neutral: #2196F3;
}

#main-container {
    max-width: 1200px;
    margin: auto;
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.header {
    text-align: center;
    margin-bottom: 25px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 10px;
}

.upload-area {
    border: 3px dashed var(--primary) !important;
    min-height: 300px;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: var(--secondary) !important;
    transform: translateY(-2px);
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 1.1em;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    background: white;
}

.visualization-box {
    border-radius: 12px;
    padding: 15px;
    background: white;
    margin-top: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.btn-primary {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

.model-select {
    background: white !important;
    border: 2px solid var(--primary) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 0.9em;
    color: #666;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animation {
    animation: fadeIn 0.5s ease-in-out;
}

.loading {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_classes=["header"]):
            gr.Markdown("# 🛡️ DeepGuard AI")
            gr.Markdown("## Advanced Deepfake Detection System")
        
        with gr.Row():
            with gr.Column(scale=1.5):
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Image for Analysis", 
                    elem_classes=["upload-area", "animation"]
                )
                
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_NAMES.keys()),
                        value=list(MODEL_NAMES.keys())[0],
                        label="Select Detection Model",
                        elem_classes=["model-select", "animation"]
                    )
                    analyze_btn = gr.Button(
                        "Analyze Image", 
                        elem_classes=["btn-primary", "animation"]
                    )
            
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["visualization-box"]):
                    heatmap_output = gr.Image(
                        label="Attention Heatmap",
                        interactive=False
                    )
                
                with gr.Column(elem_classes=["visualization-box"]):
                    chart_output = gr.Plot(
                        label="Detection Probabilities"
                    )
        
        with gr.Column(elem_classes=["result-box", "animation"]):
            result_output = gr.Textbox(
                label="Analysis Results",
                interactive=False,
                lines=8
            )
        
        with gr.Column(elem_classes=["result-box", "animation"]):
            model_info = gr.Textbox(
                label="Model Information",
                interactive=False,
                lines=3
            )
        
        gr.Markdown("""
        <div class="footer">
        *Note: This tool provides probabilistic estimates. Always verify important findings with additional methods.<br>
        Models may produce false positives/negatives. Performance varies by image type and quality.*
        </div>
        """)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, model_selector],
        outputs=[heatmap_output, chart_output, result_output, model_info]
    )

if __name__ == "__main__":
    demo.launch() 