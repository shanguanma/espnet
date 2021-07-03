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
from collections import OrderedDict


import humanfriendly
import torch
from typeguard import check_argument_types
import librosa
import numpy as np
import scipy.io.wavfile as wf

from torch_complex.tensor import ComplexTensor
import torchaudio
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh1_check import EnhancementTask
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
    output_of_stft_file: str,
    magnitude_of_stft_file: str,
    phase_of_stft_file: str,
    numpy_magnitude_of_stft_file: str,
    apply_cmvn_numpy_magnitude_of_stft_file: str,
    apply_cmvn_magnitude_of_stft_file: str,
    input_of_rnn_file: str,
    output_of_rnn_file: str,
    drop_out_file: str,
    input_of_linear_file: str,
    output_of_linear_file: str,
    mask_output_file: str,
    librosa_stft_mat_file: str,
    predicted_complex_mat_file: str,
    predicted_magnitude_mat_file: str,
    predicted_phase_mat_file: str,
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
    list_1=[]
    for keys, batch in loader:
        list_1.append(keys[0])
    print(f"list_1 is {list_1}")
    wav2scp = {}
    #path_name_type_list = copy.deepcopy(data_path_and_name_and_type)
    #logging.info(f"path_name_type_list is {path_name_type_list}") # path_name_type_list is [('dump_8k_1/stft2/data_Ach_train_8k_new_espnet2/feats.scp', 'magnitude_mix', 'npy')]

    with open(wav_scp, "r") as f:
        for line in f:
            line = line.rstrip().split(None, 1)
            wav2scp[line[0]] = line[1]
    mvn_dict="/home4/md510/w2020/espnet-recipe/enhancement_on_espnet2/data/data_Ach_train_8k_new_espnet2_2/cmvn_center_true3.dst"
    with open(mvn_dict, "rb") as f:
        dict_mvn = pickle.load(f)
    num_utts=0
    output_of_stft={}
    magnitude_of_stft={}
    phase_of_stft={}
    numpy_magnitude_of_stft={}
    apply_cmvn_numpy_magnitude_of_stft={}
    apply_cmvn_magnitude_of_stft={}
    input_of_rnn={}
    output_of_rnn={} 
    drop_out={}
    input_of_linear={}
    output_of_linear={}
    mask_output={}
    librosa_stft_mat={}
    predicted_complex_mat={}
    predicted_magnitude_mat={}
    predicted_phase_mat={}
    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        logging.info(f"keys is {keys} and batch is {batch}")
        #logging.info(f"keys is {keys} and batch is {batch} and {batch['magnitude_mix'].shape}")
        # (TODO)
        # read recorrespoding wav.scp
        #list_1.append(keys[0])              
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
            
            #predicted_spectrums, flens, masks = enh_model.enh_model.forward(
            #    batch["speech_mix"], batch["speech_mix_lengths"]
            #)
            # 1. get stft feature
            uttid= keys[0]
            input_spectrum, flens = enh_model.enh_model.stft(batch["speech_mix"], batch["speech_mix_lengths"])
            output_of_stft_1 = {uttid: input_spectrum}
            output_of_stft.update(output_of_stft_1) 
            logging.info(f"in the enh_inference1_check.py inference function, input is self.stft is {batch['speech_mix']} its shape is {batch['speech_mix'].shape}")
            logging.info(f"in the enh_inference1_check.py inference function, output is self.stft is {input_spectrum} its shape is {input_spectrum.shape}")
            input_magnitude, input_phase = torchaudio.functional.magphase(input_spectrum)
            magnitude_of_stft_1 = {uttid: input_magnitude}
            magnitude_of_stft.update(magnitude_of_stft_1)
            phase_of_stft_1 = {uttid: input_phase}
            phase_of_stft.update(phase_of_stft_1)
            logging.info(f"in the enh_inference1_check.py inference function, input magnitude is {input_magnitude}, its shape is {input_magnitude.shape}")
            logging.info(f"in the enh_inference1_check.py inference function, input phase is {input_phase} its shape is {input_phase.shape}")
            # apply apply global mvn
            input_magnitude_numpy = input_magnitude.cpu().data.numpy()
            numpy_magnitude_of_stft_1 = {uttid: input_magnitude_numpy}
            numpy_magnitude_of_stft.update(numpy_magnitude_of_stft_1)
            input_magnitude_mvn_numpy = apply_cmvn(input_magnitude_numpy, dict_mvn)
            apply_cmvn_numpy_magnitude_of_stft_1 = {uttid: input_magnitude_mvn_numpy}
            apply_cmvn_numpy_magnitude_of_stft.update(apply_cmvn_numpy_magnitude_of_stft_1)
            logging.info(f"in the enh_inference1_check.py inference function,dict_mvn is {dict_mvn}")
            logging.info(f"in the enh_inference1_check.py inference function,after global_cmvn  input_magnitude_mvn_numpy  is {input_magnitude_mvn_numpy}")
            input_magnitude_mvn = torch.tensor(input_magnitude_mvn_numpy,dtype=torch.float32,device=batch["speech_mix_lengths"].device)
            apply_cmvn_magnitude_of_stft_1 = {uttid: input_magnitude_mvn}
            apply_cmvn_magnitude_of_stft.update(apply_cmvn_magnitude_of_stft_1)
            logging.info(f"in the enh_inference1_check.py inference function,input_magnitude_mvn  device is {input_magnitude_mvn.device}")
            logging.info(f"in the enh_inference1_check.py inference function,ilens dtype is {batch['speech_mix_lengths'].device}")
            # predict masks for each speaker
            #uttid= keys[0]
            #list_1.append(uttid)
            x, flens, _ = enh_model.enh_model.rnn(input_magnitude_mvn, flens) 
            input_of_rnn1 = {uttid: input_magnitude_mvn}
            input_of_rnn.update(input_of_rnn1)
            output_of_rnn1 = {uttid: x}
            output_of_rnn.update(output_of_rnn1)

            x = enh_model.enh_model.dropout(x)
            dropout_x = {uttid: x }
            drop_out.update(dropout_x) 
            logging.info(f"in the enh_inference1_check.py inference function,output of self.drop is {x} its shape is {x.shape}")
            masks = []
            for linear in enh_model.enh_model.linear:
                y = linear(x)
                input_of_linear_1 = {uttid: x}
                input_of_linear.update(input_of_linear_1)
                output_of_linear_1 = {uttid: y}
                output_of_linear.update(output_of_linear_1)
                y = enh_model.enh_model.nonlinear(y)
                mask_output_1 = {uttid: y}
                mask_output.update(mask_output_1)
                masks.append(y)
            masks = OrderedDict(zip(["spk{}".format(i + 1) for i in range(len(masks))], masks))
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
            librosa_stft_mat_1 = {uttid: stft_mat}
            librosa_stft_mat.update(librosa_stft_mat_1)
            # b4 get enhanced wave
            #masks = masks[0].squeeze(0).cpu().data.numpy()
             
            masks = masks["spk1"].squeeze(0).cpu().data.numpy() 
            masks = masks.transpose() # freq x time 
            predicted_complex = stft_mat * masks
            predicted_complex_mat_1 = {uttid: predicted_complex}
            predicted_complex_mat.update(predicted_complex_mat_1)
            logging.info(f"predicted_complex feature is {predicted_complex} and its shape is {predicted_complex.shape}")
            predicted_magnitude = np.abs(predicted_complex)# freq x time
            predicted_magnitude_tensor = torch.tensor(predicted_magnitude,dtype=torch.float32, device=device)
            predicted_magnitude = predicted_magnitude_tensor.permute(1,0).unsqueeze(0) # 1x time x freq
            predicted_magnitude_mat_1 = {uttid: predicted_magnitude}
            predicted_magnitude_mat.update(predicted_magnitude_mat_1)
            logging.info(f"np.abs(predicted_complex) is {predicted_magnitude} and its shape is {predicted_magnitude.shape}")
            predicted_phase = np.angle(predicted_complex)
            predicted_phase_mat_1 = {uttid: predicted_phase}
            predicted_phase_mat.update(predicted_phase_mat_1)
            logging.info(f"predicted_phase is {predicted_phase} and its shape is {predicted_phase.shape}")
            # b5 
            samps = librosa.istft(
                predicted_complex, 256, 512, window="hann", center=True, length=samps.size,
            )
            logging.info(f"samps is {samps} and samps shape is {samps.shape} and its dtype is {samps.dtype}")
            samps_int16 = (samps * MAX_INT16).astype(np.int16)
            logging.info(f"samps_int16 is  {samps_int16} and its shape is {samps_int16.shape}")
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
            #print(f"list_1 is {list_1}")
    with open(output_of_stft_file, "wb") as f:
         pickle.dump(output_of_stft, f)
    with open( magnitude_of_stft_file, "wb") as f:
        pickle.dump( magnitude_of_stft, f)
    with open(phase_of_stft_file, "wb") as f:
        pickle.dump(phase_of_stft, f)
    with open(numpy_magnitude_of_stft_file, "wb") as f:
        pickle.dump(numpy_magnitude_of_stft, f)
    with open(apply_cmvn_numpy_magnitude_of_stft_file, "wb") as f:
        pickle.dump(apply_cmvn_numpy_magnitude_of_stft, f)
    with open(apply_cmvn_magnitude_of_stft_file, "wb") as f:
        pickle.dump(apply_cmvn_magnitude_of_stft, f)

    with open(input_of_rnn_file, "wb") as f:
        pickle.dump(input_of_rnn, f)
    with open(output_of_rnn_file, "wb") as f:
        pickle.dump(output_of_rnn, f) 
    with open(drop_out_file, "wb") as f:
        pickle.dump(drop_out, f)
    with open(input_of_linear_file, "wb") as f:
        pickle.dump(input_of_linear, f)
    with open(output_of_linear_file, "wb") as f:
        pickle.dump(output_of_linear, f)
    with open(mask_output_file, "wb") as f:
        pickle.dump(mask_output, f)

    with open(librosa_stft_mat_file, "wb") as f:
        pickle.dump(librosa_stft_mat, f)
    with open(predicted_complex_mat_file, "wb") as f:
        pickle.dump(predicted_complex_mat, f)
    with open(predicted_magnitude_mat_file, "wb") as f:
        pickle.dump(predicted_magnitude_mat, f)
    with open(predicted_phase_mat_file, "wb") as f:
        pickle.dump(predicted_phase_mat, f)
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
    group = parser.add_argument_group("store related interal results ")
    group.add_argument(
        "--output_of_stft_file",
        type=str,
        help="output complex stft feature of after stft layer",
    )
    group.add_argument(
        "--magnitude_of_stft_file",
        type=str,
        help="output magnitude spectrum of complex stft feature of after stft layer",
    )
    group.add_argument(
        "--phase_of_stft_file",
        type=str,
        help="output phase spectrum of complex stft feature of after stft layer",
    )
    group.add_argument(
        "--numpy_magnitude_of_stft_file",
        type=str,
        help="output numpy style magnitude spectrum of complex stft feature of after stft layer",
    )
    group.add_argument(
        "--apply_cmvn_numpy_magnitude_of_stft_file",
        type=str,
        help="output apply cmvn numpy style magnitude spectrum of complex stft feature of after stft layer",
    )
    group.add_argument(
        "--apply_cmvn_magnitude_of_stft_file",
        type=str,
        help="output apply cmvn tensor style magnitude spectrum of complex stft feature of after stft layer",
    )   
    group.add_argument(
        "--input_of_rnn_file",
        type=str,
        #default="exp_8k_1/enh_train_enh_tf_mask_magnitude3_same_data_Ach_train_8k_8k_raw/enhanced_data_Ach_train_8k_10/input_of_rnn_file.pkl",
        help="input of rnn will be stored",
    )
    group.add_argument(
        "--output_of_rnn_file",
        type=str,
        #default="exp_8k_1/enh_train_enh_tf_mask_magnitude3_same_data_Ach_train_8k_8k_raw/enhanced_data_Ach_train_8k_10/output_of_rnn_file.pkl",
        help="output of rnn will be stored",
    )
    group.add_argument(
        "--drop_out_file",
        type=str,
        #default="exp_8k_1/enh_train_enh_tf_mask_magnitude3_same_data_Ach_train_8k_8k_raw/enhanced_data_Ach_train_8k_10/drop_out_file.pkl",
        help="output of dropout layer will be stored",
    )
    group.add_argument(
        "--input_of_linear_file",
        type=str,
        help="input of linear layer will be stored",
    )
    group.add_argument(
        "--output_of_linear_file",
        type=str,
        help="output of linear layer will be stored",
    )
    group.add_argument(
        "--mask_output_file",
        type=str,
        help="output of non layer will be stored",
    )
    group.add_argument(
        "--librosa_stft_mat_file",
        type=str,
        help="librosa style stft complex matrix with noisy wavform, it is use to predicted magnitude=abs(librosa_stft_mat_file * mask)",
    )
    group.add_argument(
        "--predicted_complex_mat_file",
        type=str,
        help="predicted complex_mat , it is reconstrut enhanced wav samples, it is equal to librosa_stft_mat_file * mask",
    )
    group.add_argument(
        "--predicted_magnitude_mat_file",
        type=str,
        help="predicted magnitude_mat of complex_mat, predicted magnitude=abs(librosa_stft_mat_file * mask)",
    )
    group.add_argument(
        "--predicted_phase_mat_file",
        type=str,
        help="predicted phase mat of complex_mat, predicted phase=np.angle(librosa_stft_mat_file * mask)",
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
