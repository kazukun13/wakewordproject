import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F  # might be used later, left in

class ECAPAExportModel(nn.Module):
    def __init__(self, hdim=100, outdim=4):
        super().__init__()
        # pull just the embedding model (we don't need the full classifier)
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        ).mods.embedding_model

        self.head = nn.Sequential(
            nn.Linear(192, hdim),
            nn.ReLU(),
            nn.Linear(hdim, outdim)
        )

    def forward(self, wav):
        with torch.no_grad():
            x = self.ecapa(wav)
            if x.dim() == 3:
                x = x.mean(dim=1)
        return self.head(x)

if __name__ == "__main__":
    dummy = torch.randn(1, 16000)
    model = ECAPAExportModel()
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        "ecapa_combined.onnx",
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={"audio": {1: "time"}},
        opset_version=11
    )

    print("ONNX saved as ecapa_combined.onnx")

    traced = torch.jit.trace(model, dummy)
    traced.save("ecapa_combined.pt")
    print("TorchScript model saved as ecapa_combined.pt")
