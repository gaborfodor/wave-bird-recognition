import base64
import io

import librosa
import markdown
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objects as go, io as pio

from birds.pann import SAMPLE_RATE, N_FFT, MIN_FREQ, MAX_FREQ, HOP_SIZE, CHUNK_DURATION, read_audio_fast


def get_taxonomy():
    taxonomy = pd.read_csv('./data/taxonomy.csv')
    return taxonomy.set_index('speciesCode')


tx = get_taxonomy()


def random_chunk(y, duration, sr=SAMPLE_RATE):
    sample_size = int(duration * sr)
    if y.shape[0] < sample_size:
        raise ValueError('Too short clip')
    t = np.random.randint(0, y.shape[0] - sample_size)
    return y[t: t + sample_size]


def random_cyclic_shift(y):
    return np.roll(y, shift=np.random.randint(0, len(y)))


BIRD_INFO_MD_TEMPLATE = '''
## **{comName}**
### *{sciName}*
**Ebird code**: {ebird_code}\n
**Order**: {order}\n
**Family**: {familyComName} *({familySciName})*

<img src={img_path} alt={comName} width="400"/>
'''

AUDIO_TEMPLATE = '''
<html>
    <body>
        <audio controls src="{uploaded}">
            Your browser does not support the <code>audio</code> element.
        </audio>
    </body>
</html>
'''


async def create_bird_info(q, ebird_code):
    bird_info = tx.loc[ebird_code].to_dict()
    img_path, = await q.site.upload([f'./data/imgs/{ebird_code}.jpg'])
    bird_info['img_path'] = img_path
    bird_info['ebird_code'] = ebird_code
    return BIRD_INFO_MD_TEMPLATE.format(**bird_info)


def create_mel_spectogram(y):
    mel = librosa.feature.melspectrogram(
        y, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=128,
        sr=SAMPLE_RATE, fmin=MIN_FREQ, fmax=MAX_FREQ
    )
    return librosa.power_to_db(mel, ref=np.max)


def random_sample_spectogram(ebird):
    path = f'./data/audio/{ebird}.mp3'
    y = read_audio_fast(path)
    chunk = random_cyclic_shift(random_chunk(y, 5))
    return create_mel_spectogram(chunk)


def geo_plot(ebird_code, dx, dy):
    df = pd.read_csv(f'./data/geo/geo_{ebird_code}.csv.gz')
    fig = go.Figure(data=go.Scattergeo(
        lat=df['lat'],
        lon=df['lon'],
        marker=dict(
            color=df['cnt'],
            reversescale=True,
            opacity=0.5,
            line=dict(width=0),
            size=np.log10(df.cnt) + 1,
        ),
    ))
    _ = fig.update_layout(
        geo=dict(
            scope='north america',
            showland=True,
            landcolor="rgb(212, 212, 212)",
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            showsubunits=True,
            showcountries=True,
            resolution=50,
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[-140.0, -55.0],
                dtick=5
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[20.0, 60.0],
                dtick=5
            )
        ),
        width=dx * 134,
        height=dy * 76 - 10,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    config = {'scrollZoom': False, 'showLink': False, 'displayModeBar': False}
    return pio.to_html(fig, validate=False, include_plotlyjs='cdn', config=config)


def create_random_spectogram(ebird_code):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    plt.imshow(random_sample_spectogram(ebird_code), origin='lower', aspect='auto', cmap=plt.cm.Greys)
    plt.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image


def create_spectogram_from_audio(y, filename, fig_size):
    fig, ax = plt.subplots(figsize=fig_size)
    spec = create_mel_spectogram(y)
    plt.imshow(spec, origin='lower', aspect='auto', cmap=plt.cm.Greys)
    duration = y.shape[0] / SAMPLE_RATE
    xs = np.arange(0, duration, CHUNK_DURATION)
    frames = librosa.core.time_to_frames(xs, sr=SAMPLE_RATE, hop_length=HOP_SIZE, n_fft=N_FFT)
    ax.set_xticks(frames)
    ax.set_xticklabels([int(x) for x in xs])
    ax.set_xlim((0, spec.shape[1]))
    ax.set_yticks([])
    plt.box(on=None)
    plt.suptitle(f'Mel Spectrogram - {filename}')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image


def top_bird_bar_plot(top_birds, dx, dy):
    fig = go.Figure(go.Bar(
        x=top_birds.p,
        y=top_birds.ebird,
        marker=dict(color=top_birds.color),
        orientation='h'))
    _ = fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=dx * 134 - 10,
        height=dy * 76 - 10,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    config = {'scrollZoom': False, 'showLink': False, 'displayModeBar': False}
    return pio.to_html(fig, validate=False, include_plotlyjs='cdn', config=config)


def top_bird_line_chart(predictions, top_birds, dx, dy):
    fig = go.Figure([
        go.Scatter(
            x=predictions.start_second,
            y=predictions[bird],
            name=bird,
            opacity=0.8,
            line=dict(color=color, width=3, dash=dash),
            fill=fill,
        ) for bird, color, dash, fill in top_birds[['ebird', 'color', 'dash', 'fill']].values]
    )
    _ = fig.update_layout(
        yaxis=dict(showgrid=False, visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=dx * 134 - 10,
        height=dy * 76 - 10,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    config = {'scrollZoom': False, 'showLink': False, 'displayModeBar': False}
    return pio.to_html(fig, validate=False, include_plotlyjs='cdn', config=config)


def show_references():
    with open('birds/ref.md', 'r', encoding='utf-8') as f:
        text = f.read()
    html = markdown.markdown(text)
    return html
