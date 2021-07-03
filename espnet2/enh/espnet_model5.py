from distutils.version import LooseVersion
from functools import reduce
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple
import logging
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

# from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

"""
This part for custum time-frequency domain  style speech enhancement.
input is magnitude
output is magnitude
"""
import pysnooper

is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")
ALL_LOSS_TYPES = (
    # mse_loss(predicted_mask, target_label)
    "mask_mse",
    "mask_sumse",
    # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
    "magnitude",
    "magnitude1",
    "magnitude2",
    "magnitude3",
    # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
    "spectrum",
    # log_mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
    "spectrum_log",
    # si_snr(enhanced_waveform, target_waveform)
    "si_snr",
)
EPS = torch.finfo(torch.get_default_dtype()).eps

# @pysnooper.snoop()
class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        # encoder: AbsEncoder,
        separator: AbsSeparator,
        #decoder: AbsDecoder,
        stft_consistency: bool = False,
        loss_type: str = "mask_sumse",
        mask_type: Optional[str] = None,
    ):
        assert check_argument_types()

        super().__init__()
        # self.encoder = encoder
        self.separator = separator
        #self.decoder = decoder
        self.num_spk = separator.num_spk

        # self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # if loss_type != "si_snr" and isinstance(encoder, ConvEncoder):
        #    raise TypeError(f"{loss_type} is not supported with {type(ConvEncoder)}")

        # get mask type for TF-domain models (only used when loss_type="mask_*")
        self.mask_type = mask_type.upper() if mask_type else None
        # get loss type for model training
        self.loss_type = loss_type
        # whether to compute the TF-domain loss while enforcing STFT consistency
        self.stft_consistency = stft_consistency

        if stft_consistency and loss_type in ["mask_mse", "mask_sumse", "si_snr"]:
            raise ValueError(
                f"stft_consistency will not work when '{loss_type}' loss is used"
            )

        assert self.loss_type in ALL_LOSS_TYPES, self.loss_type
        # for multi-channel signal
        # self.ref_channel = getattr(self.separator, "ref_channel", -1)

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        :param mix_spec: Tensor(B, T, F)
        :param ref_spec: [Tensor(B, T, F), ...] or Tensor(B, T, F)
        :param noise_spec: Tensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
        ], f"mask type {mask_type} not supported"

        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [r >= n for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = r / (sum(([n for n in ref_spec])) + EPS)
            elif mask_type == "IAM":
                mask = r / (mix_spec + EPS)
                mask = mask.clamp(min=0, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
        self,
        magnitude_mix: torch.Tensor,
        magnitude_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
           feature + separator+ Decoder + Calc loss

        Args:
            magnitude_mix: (Batch, frames, frequency)
            magbitude_ref: (Batch, num_speaker, frames, frequency)
            magnitude_mix_lengths: (Batch,),
        """

        # clean speech signal of each speaker
        magnitude_ref = [
            kwargs["magnitude_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        #magnitude_ref = kwargs["magnitude_ref{}".format(1)] 
        # (Batch, num_speaker, time, frequency)
        magnitude_ref = torch.stack(magnitude_ref, dim=1)

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        # dereverb_speech_ref = kwargs.get("dereverb_ref", None)

        batch_size = magnitude_mix.shape[0]
        magnitude_lengths = (
            magnitude_mix_lengths
            if magnitude_mix_lengths is not None
            else torch.ones(batch_size).int() * magnitude_mix.shape[1]
        )
        assert magnitude_lengths.dim() == 1, magnitude_lengths.shape
        # Check that batch_size is unified
        assert (
            magnitude_mix.shape[0]
            == magnitude_ref.shape[0]
            == magnitude_lengths.shape[0]
        ), (
            magnitude_mix.shape,
            magnitude_ref.shape,
            magnitude_lengths.shape,
        )

        # for data-parallel(for Check whether it is correct or not)(TODO)
        magnitude_ref = magnitude_ref[:, :, : magnitude_mix_lengths.max()]
        magnitude_mix = magnitude_mix[:, : magnitude_mix_lengths.max()]
        logging.info(f"magnitude_ref  is {magnitude_ref.shape}")
        logging.info(f"magnitude_mix is {magnitude_mix.shape}")
        magnitude_ref = magnitude_ref.squeeze(1) 
        #logging.info(f"magnitude_ref squeeze(1)  is {magnitude_ref.shape}")
        #logging.info(f"enhanced model input is magnitude_mix is {magnitude_mix} its shape is {magnitude_mix.shape}")
        # forward network
        loss, perm = self._compute_loss(
            magnitude_mix,
            magnitude_lengths,
            magnitude_ref,
        )
        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        logging.info(f"final loss is {loss}, stats is {stats}, weight is {weight}")
        return loss, stats, weight

    def _compute_loss(
        self,
        magnitude_mix,
        magnitude_lengths,
        magnitude_ref,
    ):

        """Compute loss according to self.loss_type.
        Args:
            magnitude_mix: (Batch, frames, frequency) or (Batch, frames, frequency, channels)
            magnitude_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            magnitude_ref: (Batch, num_speaker, frames frequecy)
                        or (Batch, num_speaker, frames, frequecy, channels)


         Returns:
            loss: (torch.Tensor) speech enhancement loss
            perm: () best permutation
        """
         
        # magnitude, ilens, masks= self.separator(magnitude_mix, magnitude_lengths)

        if self.loss_type != "si_snr":

            # predict separated speech and masks
            logging.info(f"magnitude_ref  is {magnitude_ref.shape}")
            magnitude_pre, tf_length, mask_pre = self.separator(
                magnitude_mix, magnitude_lengths
            )
            logging.info(f"enhanced model  input  is {magnitude_mix} and its shape is {magnitude_mix.shape}")
            logging.info(f"enhanced model  output  is {magnitude_pre[0]} and its shape is {magnitude_pre[0].shape}")
            #logging.info(f"network output shape is {magnitude_pre[0].shape}")
            # compute TF masking loss
            if self.loss_type == "magnitude":
                # compute loss on magnitude spectrum
                magnitude_pre = [ps for ps in magnitude_pre]
                magnitude_ref = [sr for sr in magnitude_ref]
                logging.info(f"ref is {len(magnitude_ref)}")
                logging.info(f"inf is {magnitude_pre[0].shape}")
                logging.info(f"ref is {magnitude_ref[0].shape}")
                         
                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            elif self.loss_type == "magnitude1":
                # compute loss on magnitude spectrum
                #magnitude_pre = [ps for ps in magnitude_mix]
                magnitude_pre = [ps for ps in magnitude_pre]
                magnitude_ref = [sr for sr in magnitude_ref]
                logging.info(f"ref is {len(magnitude_ref)}")
                logging.info(f"inf is {magnitude_pre[0].shape}")
                logging.info(f"ref is {magnitude_ref[0].shape}")
                tf_loss, perm = self._permutation_loss1(
                #tf_loss, perm = self._permutation_loss2(
                    magnitude_ref, magnitude_pre, mask_pre
                )
            elif self.loss_type == "magnitude3":
                # compute loss on magnitude spectrum
                # magnitude_ref  is B x T x F
                # magnitude_pre[0] is B x T x F
                logging.info(f"in _compute_loss, magnitude_ref  shape is {magnitude_ref.shape}")
                logging.info(f"in _compute_loss, magnitude_pre[0]  shape is {magnitude_pre[0].shape}")
                tf_loss, perm = self._permutation_loss3(
                    magnitude_ref, magnitude_pre[0], magnitude_lengths,
                )


            elif self.loss_type == "magnitude2":
                # compute loss on magnitude spectrum
                #magnitude_pre = [ps for ps in magnitude_mix]
                magnitude_pre = [ps for ps in magnitude_pre]
                magnitude_ref = [sr for sr in magnitude_ref]
                #tf_loss, perm = self._permutation_loss1(
                tf_loss, perm = self._permutation_loss2(
                    magnitude_ref, magnitude_pre, mask_pre
                )
            elif self.loss_type.startswith("mask"):
                if self.loss_type == "mask_mse":
                    loss_func = self.tf_mse_loss
                elif self.loss_type == "mask_sumse":
                    loss_func = self.tf_sumse_loss1
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert mask_pre is not None
                mask_pre_ = [
                    mask_pre["mask_spk{}".format(spk + 1)]
                    for spk in range(self.num_spk)
                ]

                # prepare ideal masks
                mask_ref = self._create_mask_label(
                    magnitude_mix, magnitude_ref, mask_type=self.mask_type
                )

                # compute TF masking loss
                if self.loss_type == "mask_mse":
                    tf_loss, perm = self._permutation_loss(
                        mask_ref, mask_pre_, loss_func
                    )
                elif self.loss_type == "mask_sumse":
                    tf_loss, perm = loss_func(mask_ref, mask_pre_)
                    logging.info(f"mask_sumse loss is {tf_loss}")
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)
            else:
                raise ValueError("Unsupported loss type: %s" % self.loss_type)
            loss = tf_loss
            return loss, perm

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

        return mseloss

    @staticmethod
    def tf_sumse_loss1(ref, inf, perm=None):
        """time-frequency sumSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        logging.info(f"input_sizeis {ref[0].shape[0]}")
        input_size = ref[0].shape[0]
        num_spk = len(ref)

        def loss():
            loss_for_permute = []
            # logging.info(f"masks_[0]  type is {type(masks_[0])}")
            logging.info(f"ref[0] type is {type(ref[0])}")
            # N X T X F
            inf_mask = inf[0]
            #  N X T X F
            ref_mask = ref[0]
            # N X T X F
            se = torch.pow(inf_mask - ref_mask, 2)
            # N X T X 1
            se_sum_f = torch.sum(se, -1)
            # N X 1 X1
            utt_loss = torch.sum(se_sum_f, -1)
            # utt_loss = torch.sum(torch.sum(torch.pow(masks_[int(0)]*inf - ref[int(0)], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / input_size
            return loss_perutt

        logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref[0].shape[0]
        # O(N!), could be optimized
        # 1 x N
        pscore = torch.stack([loss()], dim=0)
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        logging.info(f"pscore is {pscore}")
        # N
        min_perutt, _ = torch.min(pscore, dim=0)
        logging.info(f"min_perutt is {min_perutt}")
        logging.info(f"loss is {torch.sum(min_perutt) / (num_spk * num_utts)}")
        return torch.sum(min_perutt) / (num_spk * num_utts), perm

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            l1loss = abs(ref - inf).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """si-snr loss

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """si_snr loss with zero-mean in pre-processing.

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in permutations(range(num_spk))], dim=1
        )
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        return loss.mean(), perm

    @staticmethod
    def _permutation_loss1(ref, inf, masks, perm=None):
        logging.info(f"input_sizeis {ref[0].shape[0]}")
        input_size = ref[0].shape[0]
        num_spk = len(ref)
        # logging.info(f"masks is {masks}")
        # masks_ = [
        #            masks["spk{}".format(spk + 1)] for spk in range(num_spk)
        # ]
        masks_ = [masks["mask_spk1"]]
        #logging.info(f"masks_ is {masks_}")
        # logging.info(f" masks_ type is {type(masks_)}")
        # logging.info(f"masks_ length is {len(masks_)}")
        # logging.info(f"masks_[0] is {masks_[0]}")
        logging.info(f"masks_[0] shape is {masks_[0].shape}")
        logging.info(f"ref is {ref} and inf is {inf}")
        def loss():
            loss_for_permute = []
            #logging.info(f"masks_[0]  type is {type(masks_[0])}")
            #logging.info(f"ref[0] type is {type(ref[0])}")
            # N X T X F
           
            inf_magnitude = inf[0]
            logging.info(f"inf_magnitude shape is {inf_magnitude.shape}")
            #  N X T X F
            ref_magnitude = ref[0]
            logging.info(f"ref_magnitude shape is {ref_magnitude.shape}")
            # 1 X T X F
            mse = torch.pow(inf_magnitude - ref_magnitude, 2)
            # N X T X 1
            mse_sum1 = torch.sum(mse, -1)
            # N X 1 X1
            utt_loss = torch.sum(mse_sum1, -1)
            # utt_loss = torch.sum(torch.sum(torch.pow(masks_[int(0)]*inf - ref[int(0)], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / input_size
            return loss_perutt

        logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref[0].shape[0]
        # O(N!), could be optimized
        # 1 x N
        pscore = torch.stack([loss()], dim=0)
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        logging.info(f"pscore is {pscore}")
        # N
        min_perutt, _ = torch.min(pscore, dim=0)
        return torch.sum(min_perutt) / (num_spk * num_utts), perm
        
    @staticmethod
    def _permutation_loss2(ref, inf, masks, perm=None):
        logging.info(f"input_sizeis {ref[0].shape[0]}")
        input_size = ref[0].shape[0]
        num_spk =1
        # logging.info(f"masks is {masks}")
        # masks_ = [
        #            masks["spk{}".format(spk + 1)] for spk in range(num_spk)
        # ]
        masks_ = [masks["mask_spk1"]]
        #logging.info(f"masks_ is {masks_}")
        # logging.info(f" masks_ type is {type(masks_)}")
        # logging.info(f"masks_ length is {len(masks_)}")
        # logging.info(f"masks_[0] is {masks_[0]}")
        logging.info(f"masks_[0] shape is {masks_[0].shape}")

        def loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute): 
                #logging.info(f"masks_[0]  type is {type(masks_[0])}")
                #logging.info(f"ref[0] type is {type(ref[0])}")
                # N X T X F
                #inf_magnitude = masks_[0] * inf[0]
                #  N X T X F
                #ref_magnitude = ref[0]
                # N X T X F
                #mse = torch.pow(inf_magnitude - ref_magnitude, 2)
                # N X T X 1
                #mse_sum1 = torch.sum(mse, -1)
                # N X 1 X1
                #utt_loss = torch.sum(mse_sum1, -1)
                utt_loss = torch.sum(torch.sum(torch.pow(inf[s] - ref[t], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / input_size
            return loss_perutt

        logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref[0].shape[0]
        # O(N!), could be optimized
        # 1 x N
        #pscore = torch.stack([loss(p) for p in permutations(range(self.num_spk)]))
        pscore = torch.stack([loss(p) for p in permutations(range(1))])
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        logging.info(f"pscore is {pscore}")
        # N
        min_perutt, _ = torch.min(pscore, dim=0)
        logging.info(f"min_perutt is {min_perutt}")
        loss = torch.sum(min_perutt) / (num_spk * num_utts)
        logging.info(f"in _permutation_loss2 function , loss is {loss}")
        return loss , perm
    @staticmethod
    def _permutation_loss3(ref, inf, magnitude_lengths, perm=None):
        logging.info(f"in _permutation_loss3, ref shape {ref.shape} and inf shape is {inf.shape}")
        logging.info(f"in _permutation_loss3, magnitude_lengths is {magnitude_lengths}")
        input_size = magnitude_lengths
        def loss():
            loss_for_permute = []
            #logging.info(f"masks_[0]  type is {type(masks_[0])}")
            #logging.info(f"ref[0] type is {type(ref[0])}")
            # N X T X F

            inf_magnitude = inf
            logging.info(f"in _permutation_loss3,inf_magnitude shape is {inf_magnitude.shape}")
            #  N X T X F
            ref_magnitude = ref
            logging.info(f"in _permutation_loss3,ref_magnitude shape is {ref_magnitude.shape}")
            # N X T X F
            mse = torch.pow(inf_magnitude - ref_magnitude, 2)
            # N X T X 1
            mse_sum1 = torch.sum(mse, -1)
            # N X 1 X1
            utt_loss = torch.sum(mse_sum1, -1)
            # utt_loss = torch.sum(torch.sum(torch.pow(masks_[int(0)]*inf - ref[int(0)], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / input_size
            return loss_perutt

        #logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref.shape[0] # batch size
        logging.info(f"in _permutation_loss3,num_utts is {num_utts}")
        # O(N!), could be optimized
        # 1 x N
        pscore = torch.stack([loss()], dim=0)
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        logging.info(f"pscore is {pscore}")
        # N
        num_spk=1
        min_perutt, _ = torch.min(pscore, dim=0)
        loss = torch.sum(min_perutt) / (num_spk * num_utts)
        """
        the loss sum freq and sum time ,then average on the time axis, then average on the number of utterances
        """
        return loss , perm
         
    def collect_feats(
        self, magnitude_mix: torch.Tensor, magnitude_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        magnitude_mix = magnitude_mix[:, : magnitude_mix_lengths.max()]

        feats, feats_lengths = magnitude_mix, magnitude_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
