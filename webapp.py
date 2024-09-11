import gradio as gr
import tensorflow as tf
from inference import GraphRNAFold_prediction, GraphRNAFold

def predictrna(rnaseq):
    output =  GraphRNAFold_prediction(rnaseq, GraphRNAFold)
    return output

interface = gr.Interface(
    fn=predictrna,
    inputs=gr.Textbox(lines=2, placeholder="Enter RNA Sequence Here"),
    outputs="text"
)

if __name__ == "__main__":
    interface.launch(share=False)
