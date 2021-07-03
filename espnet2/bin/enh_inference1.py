#!/usr/bin/env python3
import argparse
import logging
import sys
import copy
import os 
import pickle
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from pathlib import Path

import humanfriendly
import torch
from typeguard import check_argument_types
import librosa
import numpy as np
import scipy.io.wavfile as wf

from torch_complex.tensor import ComplexTensor

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh1 import EnhancementTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)

def apply_cmvn(feats, cmvn_dict):
    if type(cmvn_dict) != dict:
        raise TypeError("Input must be a python dictionary")
    if "mean" in cmvn_dict:
        feats = feats - cmvn_dict["mean"]
    if "std" in cmvn_dict:
        feats = feats / cmvn_dict["std"]
    return feats
def nfft(window_size):
    return int(2 ** np.ceil(int(np.log2(window_size))))
# return F x T or T x F
def stft(
    file,
    frame_length=512,
    frame_shift=256,
    center=False,
    window="hann",
    return_samps=False,
    apply_abs=False,
    apply_log=False,
    apply_pow=False,
    transpose=True, # If it is true, it T x F, otherwise F x T
):
    if not os.path.exists(file):
        raise FileNotFoundError("Input file {} do not exists!".format(file))
    if apply_log and not apply_abs:
        apply_abs = True
        warnings.warn("Ignore apply_abs=False cause function return real values")
    samps, _ = librosa.load(file) # its output sample rate is alway 22050, not is actully sample of audio file
    stft_mat = librosa.stft(
        samps,
        frame_length,
        frame_shift,
        frame_length,
        window=window,
        center=center,
    )
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat if not return_samps else (samps,stft_mat)

def istft(
    file,
    stft_mat,
    frame_length=512,
    frame_shift=256,
    center=False,
    window="hann",
    transpose=True,
    norm=None,
    fs=8000,
    nsamps=None,
):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    samps = librosa.istft(
        stft_mat, frame_shift, frame_length, window=window, center=center, length=nsamps
    )
    # renorm if needed
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / samps_norm
    # same as MATLAB and kaldi
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    fdir = os.path.dirname(file)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(file, fs, samps_int16)

def inference(
    output_dir: str,
    #dict_mvn: str,
    wav_scp: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    enh_train_config: str,
    enh_model_file: str,
    allow_variable_data_keys: bool,
    normalize_output_wav: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build Enh model
    enh_model, enh_train_args = EnhancementTask.build_model_from_file(
        enh_train_config, enh_model_file, device
    )
    enh_model.eval()

    num_spk =1

    # 3. Build data-iterator
    loader = EnhancementTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=EnhancementTask.build_preprocess_fn(enh_train_args, False),
        collate_fn=EnhancementTask.build_collate_fn(enh_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    #writers = []
    #for i in range(num_spk):
    #    writers.append(
    #        SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
    #    )
    wav2scp = {}
    #path_name_type_list = copy.deepcopy(data_path_and_name_and_type)
    #logging.info(f"path_name_type_list is {path_name_type_list}") # path_name_type_list is [('dump_8k_1/stft2/data_Ach_train_8k_new_espnet2/feats.scp', 'magnitude_mix', 'npy')]

    with open(wav_scp, "r") as f:
        for line in f:
            line = line.rstrip().split(None, 1)
            wav2scp[line[0]] = line[1]
    
    #with open(dict_mvn, "rb") as f:
    #        dict_mvn = pickle.load(f)
    num_utts=0 
    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        logging.info(f"keys is {keys} and batch is {batch}")
        #logging.info(f"keys is {keys} and batch is {batch} and {batch['magnitude_mix'].shape}")
        # (TODO)
        # read recorrespoding wav.scp
                          
        with torch.no_grad():
            # a. To device
            batch = to_device(batch, device)
            # b. Forward Enhancement
            # b1. apply cmvn 
            # b2. get enhanced feature from enh_model.separator
            # b3. get samps of mixture wave from stft
            # b4, get enhanced wavefrom  with b3 and b2
            # # apply cmvn 
            #logging.info(f"batch['magnitude_mix'] shape is  ") 
            #magnitude_mix_mvn = apply_cmvn(batch["magnitude_mix"], dict_mvn) 
            #logging.info(f"keys is {keys} and after cmvn  {magnitude_mix_mvn}") 
            #magnitude_mix_mvn = torch.tensor(magnitude_mix_mvn,dtype=torch.float32, device=device)
            # # b2. get masks from enh_model.separator.inference
            #masks = enh_model.separator.inference(
            #    magnitude_mix_mvn, batch["magnitude_mix_lengths"]
            #)
            
            predicted_spectrums, flens, masks = enh_model.enh_model.forward(
                batch["speech_mix"], batch["speech_mix_lengths"]
            )
            
            logging.info(f"masks is {masks} and masks['spk1'] is {masks['spk1']}") 
            # b3 get sample point, and complex stft feature
            # its output sample rate is alway 22050, not is actully sample of audio file, 
            # (MD)Note:it must set sample rate of input wave
            sample_rate = 8000
            samps, _ = librosa.load(wav2scp[keys[0]], sr=sample_rate)
            logging.info(f"samps is {samps} and its shape is {samps.shape}")
            stft_mat = librosa.stft(
                samps,
                512,
                256,
                512,
                window="hann",
                center=True,
            )
            logging.info(f"stft_mat is {stft_mat} its shape is {stft_mat.shape}")
            # b4 get enhanced wave
            #masks = masks[0].squeeze(0).cpu().data.numpy()
             
            masks = masks["spk1"].squeeze(0).cpu().data.numpy() 
            masks = masks.transpose() # freq x time 
            predicted_complex = stft_mat * masks
            logging.info(f"predicted_complex feature is {predicted_complex} and its shape is {predicted_complex.shape}")
            predicted_magnitude = np.abs(predicted_complex)
            logging.info(f"np.abs(predicted_complex) is {predicted_magnitude} and its shape is {predicted_magnitude.shape}")
            predicted_phase = np.angle(predicted_complex)
            logging.info(f"predicted_phase is {predicted_phase} and its shape is {predicted_phase.shape}")
            # b5 
            samps = librosa.istft(
                predicted_complex, 256, 512, window="hann", center=True, length=samps.size,
            )
            logging.info(f"samps is {samps} and samps shape is {samps.shape} and its dtype is {samps.dtype}")
            samps_int16 = (samps * MAX_INT16).astype(np.int16)
            #import soundfile as sf
            #sf.write('enh_test_3.wav', samps_int16, 8000) 
            import scipy.io.wavfile as wf
            file = Path(f"{output_dir}/wavs_center_true3")
            fdir = os.path.dirname(file)
            if fdir and not os.path.exists(fdir):
                os.makedirs(fdir)
            wf.write(
                os.path.join(fdir, "{}.wav".format(keys[0])),
                8000,
                samps_int16,
            )
 
            logging.info(f"Processing {num_utts} utterance {keys[0]}") 
 
def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    #parser.add_argument("--dict_mvn",type=str, help="global cmvn file")
    parser.add_argument("--wav_scp", type=str, help="wav.scp file path")
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Output data related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Weather to normalize the predicted wav to [-1~1]",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--enh_train_config", type=str, required=True)
    group.add_argument("--enh_model_file", type=str, required=True)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    return parser


def test_case():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav'   
    # 1. get magnitude feature of stft
    _, mixture_magnitude_feature= stft(
                wave_file,
                frame_length=512,
                frame_shift=256,
                window="hann",
                center=True,
                return_samps=True,
                transpose=True,
                apply_abs=True,
    )
    dict_mvn="data/data_Ach_train_8k_new_espnet2_1/cmvn_center_true.dst"
    with open(dict_mvn, "rb") as f:
            dict_mvn = pickle.load(f)
    # 2. apply gloabl cmvn
    logging.warning(f"mixture_magnitude_feature is {mixture_magnitude_feature} and its shape is {mixture_magnitude_feature.shape}")
    magnitude_mix_mvn = apply_cmvn(mixture_magnitude_feature, dict_mvn)
    logging.warning(f" after cmvn  {magnitude_mix_mvn} and its shape is {magnitude_mix_mvn.shape}")
    magnitude_mix_mvn = torch.tensor(magnitude_mix_mvn,dtype=torch.float32)
    # 3. get mask 
    # 2. Build Enh model
    enh_train_config= 'exp_8k_1/enh_train_enh_tf_mask_magnitude2_new1_data_Ach_train_8k_new_espnet2_1_8000_stft3/config.yaml' 
    enh_model_file='exp_8k_1/enh_train_enh_tf_mask_magnitude2_new1_data_Ach_train_8k_new_espnet2_1_8000_stft3/valid.loss.ave_1best.pth'
    enh_model, enh_train_args = EnhancementTask1.build_model_from_file(
        enh_train_config, enh_model_file, device = "cpu",
    )
    enh_model.eval()
    
    magnitude_mix_mvn = magnitude_mix_mvn.unsqueeze(0)
    batch_size = magnitude_mix_mvn.shape[0]
    magnitude_mix_length = torch.ones(batch_size).int() * magnitude_mix_mvn.shape[1]
    masks = enh_model.separator.inference(
        magnitude_mix_mvn, magnitude_mix_length, 
    )
    # 4. get noisy stft feature and its sample point
    samps, mixture_stft_feature= stft(
                wave_file,
                frame_length=512,
                frame_shift=256,
                window="hann",
                center=True,
                return_samps=True,
                transpose=True,
                apply_abs=False,
    )
    logging.warning(f"samps is {samps} and its shape is {samps.size}")
    masks = masks[0].squeeze(0).cpu().data.numpy()
    logging.warning(f"output mask is {masks}")
    norm = np.linalg.norm(samps, np.inf)
    # check it
    #predicted_complex = mixture_stft_feature * masks
    predicted_complex = mixture_stft_feature
    logging.warning(f"predicted_complex feature is {predicted_complex}")
    predicted_complex = np.transpose(predicted_complex)
    #istft(
          #os.path.join(Path(f"{output_dir}/wavs_center_false"), "{}.spk{}.wav".format(keys[0], 1)),
    #      "enh_test_1.wav",
    #      predicted_complex,
    #      frame_length=512,
    #      frame_shift=256,
    #      window="hann",
    #      center=True,
    #      norm=norm,
    #      fs=8000,
    ##      nsamps=samps.size,
    #     transpose=True,
    #)
    #logging.warning(f"Processing {num_utts} utterance {keys[0]}")
def test_stft_istft():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav' 
    samps, _ = librosa.load(wave_file, sr=8000) # its output sample rate is alway 22050, not is actully sample of audio file
    stft_mat = librosa.stft(
        samps,
        512,
        256,
        512,
        window="hann",
        center=True,
    )
    logging.warning(f"stft_mat shape is {stft_mat.shape}, and stft_mat is {stft_mat}")
    samps = librosa.istft(
        stft_mat, 256, 512, window="hann", center=True, length=samps.size,
    )
    logging.warning(f"samps is {samps} and samps shape is {samps.shape} and its dtype is {samps.dtype}")
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    #import soundfile as sf
    #sf.write('enh_test_3.wav', samps_int16, 8000) 
    import scipy.io.wavfile as wf
    wf.write('enh_test_3_1.wav', 8000, samps_int16,) 



def test_stft_istft1():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav'
    #samps, _ = librosa.load(wave_file, sr=8000) # its output sample rate is alway 22050, not is actully sample of audio file
    samps, _ = librosa.load(wave_file,) # its output sample rate is alway 22050, not is actully sample of audio file
    stft_mat = librosa.stft(
        samps,
        512,
        256,
        512,
        window="hann",
        center=True,
    )
    logging.warning(f"stft_mat shape is {stft_mat.shape}, and stft_mat is {stft_mat}")
    samps = librosa.istft(
        stft_mat, 256, 512, window="hann", center=True, length=samps.size,
    )
    logging.warning(f"samps is {samps} and samps shape is {samps.shape} and its dtype is {samps.dtype}")
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    #import soundfile as sf
    #sf.write('enh_test_3.wav', samps_int16, 8000) 
    import scipy.io.wavfile as wf
    wf.write('enh_test_3_2.wav', 8000, samps_int16,)
     

def test_torch_stft():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav'
    import torchaudio
    wave_, sample_rate  = torchaudio.load(wave_file)
    window = torch.hann_window(512)
    stft_mat = torch.stft(wave_, n_fft=512, hop_length=256, win_length=512, window=window, center=True,)
    logging.warning(f"stft_mat shape is {stft_mat.shape}")
     
    istft = torchaudio.functional.istft
    a = istft(stft_mat, n_fft=512,hop_length=256, win_length=512, window=window, center=True,) 
    logging.warning(f"enhanced_wav is {a}, its shape is {a.shape}")
    b = a.cpu().numpy()
    logging.warning(f"b is {b} and its shape is {b.shape}")
    import soundfile as sf
    c = np.random.rand(1,200)
    sf.write('test1.wav', c, 8000)
    #wav = "enh_test_4.wav"
    #sf.write('enh4.wav', b, 8000) # it doesn't work.




def test_torch_stft1():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav'
    import torchaudio
    wave_, sample_rate  = torchaudio.load(wave_file)
    window = torch.hann_window(512)
    stft_mat = torch.stft(wave_, n_fft=512, hop_length=256, win_length=512, window=window, center=True,)
    logging.warning(f"stft_mat shape is {stft_mat.shape}")

    istft = torch.istft
    a = istft(stft_mat, n_fft=512,hop_length=256, win_length=512, window=window, center=True,)
    logging.warning(f"enhanced_wav is {a}, its shape is {a.shape}")
    b = a.cpu().numpy()
    logging.warning(f"b is {b} and its shape is {b.shape}")
    import soundfile as sf
    c = np.random.rand(1,200)
    #sf.write('test1.wav', c, 8000)
    #wav = "enh_test_4.wav"
    path='/home4/md510/w2020/espnet-recipe/enhancement_on_espnet2/enh__5.wav'
    sf.write(path, b, 8000) 


def test_stft_istft_16k():
    wave_file='/home/nana511/maison2/Delivery2-092020/maison2/egs/check_full_real_rats_Ach/format_data/data_src_test_8k_new/data/wav/format.1/fe_03_11653-02149-B-000511-000688-A.wav'
    samps, _ = librosa.load(wave_file) # its output sample rate is alway 22050, not is actully sample of audio file
    stft_mat = librosa.stft(
        samps,
        512,
        256,
        512,
        window="hann",
        center=True,
    )
    samps = librosa.istft(
        stft_mat, 256, 512, window="hann", center=True, length=samps.size,
    )
    logging.warning(f"samps is {samps} and samps shape is {samps.shape} and its dtype is {samps.dtype}")
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    #import soundfile as sf
    #sf.write('enh_test_3.wav', samps_int16, 8000) 
    import scipy.io.wavfile as wf
    wf.write('enh_test_3_1.wav', 8000, samps_int16,)




def reconstruct8k(enhan_spec, noisy_file):
    fft_len_8k, frame_shift_8k = 256, 128
    rate, sig, nb_bits = audioread(noisy_file)
    frames = framesig(sig, fft_len_8k, frame_shift_8k, lambda x: normhamming(x), True)
    tmp = magspec(frames, fft_len_8k)
    phase_noisy = np.angle(tmp)

    # if FLAGS.feat_type.lower() == 'logmagspec':
    #     enhan_spec = np.power(10, enhan_spec / 20)
    itorch_log = np.exp(enhan_spec)
    # print('inverse log:', itorch_log)
    ipower = np.sqrt(fft_len_8k * itorch_log)
    # print('inverse power:', ipower)

    spec_comp = ipower * np.exp(phase_noisy * 1j)
    enhan_frames = np.fft.irfft(spec_comp)
    enhan_sig = deframesig(enhan_frames, len(sig), fft_len_8k, frame_shift_8k, lambda x: normhamming(x))
    # print('deframe:', enhan_sig)

    enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * np.max(np.abs(sig))
    max_nb_bit = float(2 ** (nb_bits - 1))
    enhan_sig = enhan_sig * (max_nb_bit - 1.0)

    if nb_bits == 16:
        enhan_sig = enhan_sig.astype(np.int16)
    elif nb_bits == 32:
        enhan_sig = enhan_sig.astype(np.int32)

    # print('enhan_sig:', enhan_sig)
    return enhan_sig, rate

import decimal
import logging
import math
import numpy


import scipy.io.wavfile as wav
import librosa
import numpy as np

import numpy
from scipy.signal import hamming


def normhamming(fft_len):
    if fft_len == 512:
        frame_shift = 160
    elif fft_len == 256:
        frame_shift = 128
    else:
        print("Wrong fft_len, current only support 16k/8k sampling rate wav")
        exit(1)
    win = numpy.sqrt(hamming(fft_len, False))
    win = win/numpy.sqrt(numpy.sum(numpy.power(win[0:fft_len:frame_shift],2)))
    return win

def audioread(filename):
    (rate, sig) = wav.read(filename)
    # print('just read:', sig)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    else:
        print('no type match!', sig.dtype)
    
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig / (max_nb_bit + 1.0)
    # nb_bits =1

    # sig, rate = librosa.load(filename, None, mono=True, offset=0.0, dtype=np.float32)
    # print('just read:', sig)
    # nb_bits = 1
    return rate, sig, nb_bits

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    # return numpy.angle(complex_spec), numpy.absolute(complex_spec)
    # modify based on xiaoxiong
    return complex_spec


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    # return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))
    # modify based on xiaoxiong
    complex_spec = magspec(frames, NFFT)
    return 1.0 / NFFT * (complex_spec * numpy.conj(complex_spec))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

def test_stft_istft_8k():
    pass 


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)
    #test_case(**kwargs)
    #test_stft_istft() 
    #test_stft_istft()
    #test_stft_istft1()
   
    #test_case()
    #test_stft_istft()
    #test_torch_stft()
    #test_torch_stft1()
if __name__ == "__main__":
    main()
