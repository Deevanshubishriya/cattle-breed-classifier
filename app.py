import gradio as gr
import tensorflow as tf
import numpy as np
import json

# 1. Load the labels
with open("breed_labels.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# 2. Load the trained model
model = tf.keras.models.load_model("cattle_breed_model.keras")
IMAGE_SIZE = (224, 224)

# 3. Prediction function
def predict_breed(image):
    if image is None:
        return {"No image": 1.0}, "Upload a cattle photo to get a prediction."

    # Resize and format the image for the model
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    
    # Make prediction
    predictions = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]
    
    # Format results
    result = {class_names[index]: float(predictions[index]) for index in top_indices}
    best_label = class_names[top_indices[0]]
    best_score = float(predictions[top_indices[0]])
    message = f"Top match: {best_label} ({best_score:.1%} confidence)"
    
    return result, message

# 4. Build the Gradio UI
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🐄 Cattle Breed Classifier")
    gr.Markdown("Upload an image of a cow, and the AI will predict its breed.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Photo")
            submit_btn = gr.Button("Classify")
        with gr.Column():
            prediction_output = gr.Label(num_top_classes=3, label="Top 3 Predictions")
            console_output = gr.Textbox(label="Result", interactive=False)
            
    submit_btn.click(
        fn=predict_breed,
        inputs=image_input,
        outputs=[prediction_output, console_output]
    )

demo.launch()