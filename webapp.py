import gradio as gr
import tensorflow as tf
from inference import GCNfold_prediction, GCNfold

def predictrna(rnaseq):
    output =  GCNfold_prediction(rnaseq, GCNfold)
    return output

interface = gr.Interface(
    fn=predictrna,
    inputs=gr.Textbox(lines=2, placeholder="Enter RNA Sequence Here"),
    outputs="text"
)

if __name__ == "__main__":
    interface.launch(share=False)
