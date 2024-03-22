import gradio as gr

from config import SERVER_PORT
from dataloader.ptbdb_loader import dataloader, example_loader
from explanation import explain


def main():
    with gr.Blocks(title="Explainable ECG Anomaly Detection").queue() as demo:
        gr.Markdown("""# Explainable ECG Anomaly Detection""")

        X, y = dataloader()
        examples = example_loader(X, y)

        with gr.Row():
            with gr.Column():
                index = gr.Textbox(label="Index", visible=False)
                signal_plot = gr.LinePlot(label="ECG Signal",
                                          x="x", y="y",
                                          x_title="Time (ms)", y_title="Amplitude (mV)")
                gt = gr.Textbox(label="Ground Truth", visible=False)
                gr.Examples(label="ECG Samples",
                            examples=examples,
                            inputs=[index, signal_plot, gt],
                            examples_per_page=10)
                clf_btn = gr.Button("Run classification")

            with gr.Column():
                pred = gr.Label(label="Prediction")
                xai = gr.Plot(label="Explanation")

            clf_btn.click(fn=explain, inputs=[index], outputs=[xai, pred])

        demo.launch(share=False, server_port=SERVER_PORT)


if __name__ == "__main__":
    main()
