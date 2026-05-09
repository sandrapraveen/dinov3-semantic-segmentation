import torch
import torch.nn as nn
import torch.nn.functional as F

from src.decoders.simple import SimpleDecoder
from src.decoders.dpt_head import DPTHead   # ← import your DPT


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        repo_dir="dinov3",
        model_name="dinov3_vitl16",
        weights=None,
        fine_tune=False,
        num_layers=1   # ← NEW
    ):
        super().__init__()

        self.model = torch.hub.load(
            repo_dir,
            model_name,
            source='local',
            pretrained=False
        )

        if weights is not None:
            print("Loading local weights:", weights)
            state_dict = torch.load(weights, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)

        for param in self.model.parameters():
            param.requires_grad = fine_tune

        self.embed_dim = self.model.norm.normalized_shape[0]

        self.num_layers = num_layers   # ← NEW

    def forward(self, x):
        features = self.model.get_intermediate_layers(
            x,
            n=self.num_layers,   # ← IMPORTANT
            reshape=True,
            return_class_token=False,
            norm=True
        )

        return features   # ← always return list
    
    

# class DinoV3Backbone(nn.Module):
#     def __init__(
#         self,
#         repo_dir="dinov3",
#         model_name="dinov3_vitl16",
#         weights=None,
#         fine_tune=False
#     ):
#         super().__init__()

#         # --- load backbone WITHOUT downloading ---
#         self.model = torch.hub.load(
#             repo_dir,
#             model_name,
#             source='local',
#             pretrained=False
#         )

#         # --- load local weights ---
#         if weights is not None:
#             print("Loading local weights:", weights)
#             state_dict = torch.load(weights, map_location="cpu")

#             # some checkpoints store weights under "model"
#             if "model" in state_dict:
#                 state_dict = state_dict["model"]

#             self.model.load_state_dict(state_dict, strict=False)

#         # --- freeze or finetune ---
#         for param in self.model.parameters():
#             param.requires_grad = fine_tune

#         # --- feature dimension ---
#         self.embed_dim = self.model.norm.normalized_shape[0]

#     def forward(self, x):
#         features = self.model.get_intermediate_layers(
#             x,
#             n=1,
#             reshape=True,
#             return_class_token=False,
#             norm=True
#         )[0]

#         return features



class DinoSegmentationModel(nn.Module):
    def __init__(
        self,
        num_classes=7,
        weights=None,
        repo_dir="dinov3",
        model_name="dinov3_vitl16",
        fine_tune=False,
        decoder_type="simple"   # ← NEW
    ):
        super().__init__()

        self.decoder_type = decoder_type

        # --- backbone ---
        self.backbone = DinoV3Backbone(
            repo_dir=repo_dir,
            model_name=model_name,
            weights=weights,
            fine_tune=fine_tune,
            num_layers=4 if decoder_type == "dpt" else 1   # ← KEY
        )

        # --- decoders ---
        self.simple_decoder = SimpleDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes
        )

        self.dpt_decoder = DPTHead(
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.backbone(x)

        if self.decoder_type == "simple":
            # features = [f4]
            out = self.simple_decoder(features[0])

        elif self.decoder_type == "dpt":
            # features = [f1, f2, f3, f4]
            out = self.dpt_decoder(features)

        # --- upsample ---
        out = F.interpolate(
            out,
            size=x.shape[-2:],   # safer than scale_factor
            mode='bilinear',
            align_corners=False
        )

        return out
    

# class DinoSegmentationModel(nn.Module):
#     def __init__(
#         self,
#         num_classes=7,
#         weights=None,
#         repo_dir="dinov3",
#         model_name="dinov3_vitl16",
#         fine_tune=False
#     ):
#         super().__init__()

#         # --- backbone ---
#         self.backbone = DinoV3Backbone(
#             repo_dir=repo_dir,
#             model_name=model_name,
#             weights=weights,
#             fine_tune=fine_tune
#         )

#         # --- decoder ---
#         self.decoder = SimpleDecoder(
#             in_channels=self.backbone.embed_dim,
#             num_classes=num_classes
#         )

#     def forward(self, x):
#         # --- feature extraction ---
#         features = self.backbone(x)

#         # --- decode ---
#         out = self.decoder(features)

#         # --- upsample to original image size ---
#         out = F.interpolate(
#             out,
#             scale_factor=16,
#             mode='bilinear',
#             align_corners=False
#         )

#         return out