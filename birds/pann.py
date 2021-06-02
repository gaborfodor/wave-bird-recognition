import os
import urllib.request

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

SAMPLE_RATE = 32000
CHUNK_DURATION = 5
N_FFT = 1536
HOP_SIZE = 320
N_MELS = 128
MIN_FREQ = 300
MAX_FREQ = 11000
MODEL_URI = 'https://s3.amazonaws.com/data.h2o.ai/qbirds/models_CNN14_CV1_C262_M1_N0_7028_15123910.pth'

BIRDS = [
    'aldfly', 'ameavo', 'amebit', 'amecro', 'amegfi', 'amekes', 'amepip', 'amered', 'amerob', 'amewig', 'amewoo',
    'amtspa', 'annhum', 'astfly', 'baisan', 'baleag', 'balori', 'banswa', 'barswa', 'bawwar', 'belkin1', 'belspa2',
    'bewwre', 'bkbcuc', 'bkbmag1', 'bkbwar', 'bkcchi', 'bkchum', 'bkhgro', 'bkpwar', 'bktspa', 'blkpho', 'blugrb1',
    'blujay', 'bnhcow', 'boboli', 'bongul', 'brdowl', 'brebla', 'brespa', 'brncre', 'brnthr', 'brthum', 'brwhaw',
    'btbwar', 'btnwar', 'btywar', 'buffle', 'buggna', 'buhvir', 'bulori', 'bushti', 'buwtea', 'buwwar', 'cacwre',
    'calgul', 'calqua', 'camwar', 'cangoo', 'canwar', 'canwre', 'carwre', 'casfin', 'caster1', 'casvir', 'cedwax',
    'chispa', 'chiswi', 'chswar', 'chukar', 'clanut', 'cliswa', 'comgol', 'comgra', 'comloo', 'commer', 'comnig',
    'comrav', 'comred', 'comter', 'comyel', 'coohaw', 'coshum', 'cowscj1', 'daejun', 'doccor', 'dowwoo', 'dusfly',
    'eargre', 'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eucdov', 'eursta', 'evegro', 'fiespa',
    'fiscro', 'foxspa', 'gadwal', 'gcrfin', 'gnttow', 'gockin', 'gocspa', 'goleag', 'grbher3', 'grcfly', 'greegr',
    'greroa', 'greyel', 'grhowl', 'grnher', 'grtgra', 'grycat', 'gryfly', 'haiwoo', 'hamfly', 'herthr', 'hoomer',
    'hoowar', 'horgre', 'horlar', 'houfin', 'houspa', 'houwre', 'indbun', 'juntit1', 'killde', 'labwoo', 'larspa',
    'lazbun', 'leabit', 'leafly', 'leasan', 'lecthr', 'lesgol', 'lesnig', 'lesyel', 'lewwoo', 'linspa', 'lobcur',
    'lobdow', 'logshr', 'lotduc', 'louwat', 'macwar', 'magwar', 'mallar3', 'marwre', 'merlin', 'moublu', 'mouchi',
    'moudov', 'norcar', 'norfli', 'norhar2', 'normoc', 'norpar', 'norpin', 'norsho', 'norwat', 'nrwswa', 'nutwoo',
    'olsfly', 'orcwar', 'osprey', 'ovenbi1', 'palwar', 'pasfly', 'pecsan', 'perfal', 'phaino', 'pibgre', 'pilwoo',
    'pingro', 'pinjay', 'pinsis', 'pinwar', 'plsvir', 'prawar', 'purfin', 'pygnut', 'rebmer', 'rebnut', 'rebsap',
    'rebwoo', 'redcro', 'redhea', 'reevir1', 'renpha', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinduc', 'robgro',
    'rocpig', 'rocwre', 'rthhum', 'ruckin', 'rudduc', 'rufgro', 'rufhum', 'rusbla', 'sagspa1', 'sagthr', 'savspa',
    'saypho', 'scatan', 'scoori', 'semplo', 'semsan', 'sheowl', 'shshaw', 'snobun', 'snogoo', 'solsan', 'sonspa',
    'sora', 'sposan', 'spotow', 'stejay', 'swahaw', 'swaspa', 'swathr', 'treswa', 'truswa', 'tuftit', 'tunswa',
    'veery', 'vesspa', 'vigswa', 'warvir', 'wesblu', 'wesgre', 'weskin', 'wesmea', 'wessan', 'westan', 'wewpew',
    'whbnut', 'whcspa', 'whfibi', 'whtspa', 'whtswi', 'wilfly', 'wilsni1', 'wiltur', 'winwre3', 'wlswar', 'wooduc',
    'wooscj2', 'woothr', 'y00475', 'yebfly', 'yebsap', 'yehbla', 'yelwar', 'yerwar', 'yetvir'
]


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.dataset_mean = 0.
        self.dataset_std = 1.
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=32, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = (x - self.dataset_mean) / self.dataset_std
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


def my_cnn14(n_fft, n_mels, n_classes=100, hop_size=320, fmin=160, fmax=10300):
    _model_config = {
        'sample_rate': 32000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000,
        'classes_num': 527
    }
    model = Cnn14(**_model_config)
    model.fc_audioset = nn.Linear(2048, n_classes, bias=True)
    init_layer(model.fc_audioset)
    model.spectrogram_extractor = Spectrogram(
        n_fft=n_fft, hop_length=hop_size, win_length=n_fft
    )
    model.logmel_extractor = LogmelFilterBank(
        sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    model.bn0 = nn.BatchNorm2d(n_mels)
    init_bn(model.bn0)
    return model


def load_pretrained_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = my_cnn14(n_fft=N_FFT, n_mels=N_MELS, hop_size=HOP_SIZE, n_classes=len(BIRDS), fmin=MIN_FREQ, fmax=MAX_FREQ)
    model_path = './models/cnn14.pth'
    if not os.path.exists(model_path):
        print('Downloading model')
        urllib.request.urlretrieve(MODEL_URI, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    _ = model.eval()
    return model


def read_audio_fast(path, duration=30):
    clip, sr_native = librosa.core.audio.__audioread_load(
        path, offset=0.0, duration=duration, dtype=np.float32)
    clip = librosa.to_mono(clip)
    if sr_native > 0:
        clip = librosa.resample(clip, sr_native, SAMPLE_RATE, res_type='kaiser_fast')
    return clip


def get_model_predictions_for_clip(y, model):
    duration = y.shape[0] // SAMPLE_RATE
    batch = []
    start_seconds = []
    for start in range(0, duration - CHUNK_DURATION + 1, 2):
        end = start + 5
        start_seconds.append(start)
        chunk = y[start * SAMPLE_RATE: end * SAMPLE_RATE]
        if len(chunk) != CHUNK_DURATION * SAMPLE_RATE:
            print(chunk.shape)
            break
        batch.append(chunk)
    batch = np.asarray(batch)
    tensors = torch.from_numpy(batch)
    with torch.no_grad():
        preds = model(tensors)['clipwise_output']
    test_preds = preds.cpu().numpy()
    pred_df = pd.DataFrame(test_preds, columns=BIRDS)
    pred_df['start_second'] = start_seconds
    return pred_df
