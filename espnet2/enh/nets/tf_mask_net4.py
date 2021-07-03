from collections import OrderedDict
from typing import Tuple
import logging
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.enh.layers.activation import Mish
import torch
from torch_complex.tensor import ComplexTensor


class TFMaskingNet4(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
        self,
        n_fft: int = 256,
        win_length: int = None,
        hop_length: int = 128,
        rnn_type: str = "blstm",
        layer: int = 3,
        unit: int = 896,
        dropout: float = 0.5,
        num_spk: int = 1,
        nonlinear: str = "relu",
        utt_mvn: bool = False,
        mask_type: str = "logNPSM",
        loss_type: str = "mask_mse",
    ):
        super(TFMaskingNet4, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        if loss_type not in (
            "mask_mse",
            "mask_sumse",
            "magnitude",
            #"magnitude1",
            "spectrum",
        ):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        if utt_mvn:
            self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)

        else:
            self.utt_mvn = None

        self.rnn = RNN(
            idim=self.num_bin,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, self.num_bin) for _ in range(self.num_spk)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh", "mish"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "mish": Mish(),
        }[nonlinear]

        if self.mask_type=="logNPSM":
            self.output_spectrum = False  
    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            separated (list[ComplexTensor]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)
        input_phase = input_spectrum / (input_magnitude + 10e-12)

        # apply utt mvn
        if self.utt_mvn:
            input_magnitude_mvn, flens = self.utt_mvn(input_magnitude, flens)
        else:
            input_magnitude_mvn = input_magnitude

        # predict masks for each speaker
        x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)
        
        #if self.training and self.loss_type.startswith("mask"):
        #    predicted_spectrums = None
        #else:
            # apply mask
            #aplly_soomth_mask=True
            #if aplly_soomth_mask:
            #    masks[0][masks[0]<=0] = 0.00001
        #    predict_magnitude = [input_magnitude * m for m in masks]
        #    predicted_spectrums = [input_phase * pm for pm in predict_magnitude]

        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        return  masks, flens, input_phase, input_magnitude

    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms.  (TODO) to check it .

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech [Batch, num_speaker, sample]
            output lengths
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # predict spectrum for each speaker
        # Cation: MD
        # ilens is sample point of input
        # flens is frames of stft feature of input
        masks, flens, mix_phase,mix_magnitude  = self.forward(input, ilens)
        
        logging.info(f"in the forward_rawwav function , masks is {masks}")
        mask_pre_ = masks["spk1"]
        logging.info(f"in the forward_rawwav function ,masks['spk1'] is {masks['spk1']} and its shape is {masks['spk1'].shape}")
        mask_pre_ = mask_pre_ - 5
        logging.info(f"in the forward_rawwav function , mask_pre_ - 5 is {mask_pre_} an its shape is {mask_pre_.shape}")
        mask_pre_1 = torch.pow(10, mask_pre_)
        logging.info(f"in the forward_rawwav function ,torch.pow(10, mask_pre_) is {torch.pow(10, mask_pre_)} its shape is {torch.pow(10, mask_pre_).shape}")
        logging.info(f"in the forward_rawwav function , mask_pre_1 max value is {torch.max(mask_pre_1)} and  mask_pre_1 min value is {torch.min(mask_pre_1)} ")
        enhanced_magnitude = [mix_magnitude * mask_pre_1]
        predicted_pre = [mix_phase * pm for pm in  enhanced_magnitude]
        logging.info(f"predicted_pre is {predicted_pre} and its shape is {predicted_pre[0].shape}") 
        if isinstance(predicted_pre, list):
            # multi-speaker input
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_pre
            ]
            logging.info(f" in the multi-speaker input predicted_wavs is {predicted_wavs} and its shape is {predicted_wavs[0].shape}")
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(predicted_pre, ilens)[0]
            logging.info(f" in the single-speaker input predicted_wavs is {predicted_wavs} and its shape is {predicted_wavs.shape}")
        return predicted_wavs, ilens, masks
