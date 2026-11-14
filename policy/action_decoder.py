from policy.diffusion import DiffusionUNetPolicy


def get_action_decoder(action_decoder_type: str):
    if action_decoder_type != "diffusion_unet":
        raise ValueError("MATE component policies require diffusion_unet action decoders.")
    return DiffusionUNetPolicy
