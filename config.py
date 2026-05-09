CONFIG = {
    # --- experiment setup ---
    "decoder": "dpt",
    "fine_tune": False,
    "use_boundary_loss": False,
    "boundary_weight": 0.1,

    # --- data ---
    "data_path": r"C:\Users\sandr\Downloads\dinooooo\LoveDA",
    "image_size": 384,
    "num_classes": 7,

    # --- training ---
    "batch_size": 2,
    "epochs": 1,
    "lr": 1e-4,

    # --- model ---
    "weights_path": "weights/dinov3_vitl16_pretrain_sat493m.pth",

    # --- reproducibility ---
    "seed": 42,

    # --- optional ---
    "device": "cuda",
    "save_every": 1,

    # normalization
    "mean": [0.430, 0.411, 0.296],
    "std":  [0.213, 0.156, 0.143],
}