import torch
import torch.nn as nn

from codes_source.Encoder import Encoder
from codes_source.Decoder import Decoder


class TrainedFaderWrapper(nn.Module):
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, attr_name: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attr = [attr_name]   

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _adapt_y(self, y: torch.Tensor) -> torch.Tensor:
         
         
        if not torch.is_tensor(y):
            y = torch.tensor(y, device=next(self.parameters()).device, dtype=torch.float32)

        y = y.float()

        # Si y est one-hot (1,2), on extrait "a" = y[:,1]
        if y.ndim == 2 and y.shape[1] == 2:
            a = y[:, 1:2]  # (B,1)
            return y, a

        # Si déjà scalaire (B,1) ou (B,)
        if y.ndim == 2 and y.shape[1] == 1:
            a = y
            # on reconstruit one-hot au besoin
            y_oh = torch.cat([1 - a, a], dim=1)
            return y_oh, a

        if y.ndim == 1:
            a = y.view(-1, 1)
            y_oh = torch.cat([1 - a, a], dim=1)
            return y_oh, a

        # fallback
        a = y.view(y.shape[0], -1)
        if a.shape[1] != 1:
            a = a[:, :1]
        y_oh = torch.cat([1 - a, a], dim=1)
        return y_oh, a

    def decode(self, z: torch.Tensor, y: torch.Tensor):
         
        y_oh, a = self._adapt_y(y)

         
        try:
            out = self.decoder(z, y_oh)
            return [out]
        except TypeError:
            pass
        except RuntimeError:
            pass

        
        out = self.decoder(z, a)
        return [out]


def load_trained_fader(model_pth: str, device: str, attr_name: str = "Male", n_attr: int = 1) -> TrainedFaderWrapper:
    """
    Charge l'export .pth qui contient:
      ckpt["encoder"] state_dict
      ckpt["decoder"] state_dict

    Retourne un objet ae  
    """
    ckpt = torch.load(model_pth, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "encoder" in ckpt and "decoder" in ckpt):
        raise ValueError(
            
        )

    encoder = Encoder().to(device)
    decoder = Decoder(n_attr=n_attr).to(device)

    encoder.load_state_dict(ckpt["encoder"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    ae = TrainedFaderWrapper(encoder, decoder, attr_name=attr_name).to(device).eval()
    return ae
