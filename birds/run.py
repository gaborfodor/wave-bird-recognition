import pandas as pd
from h2o_wave import ui, Q, data, app, main

from birds.display_utils import geo_plot, create_random_spectogram, create_bird_info, create_spectogram_from_audio, \
    top_bird_bar_plot, top_bird_line_chart, tx, show_references, AUDIO_TEMPLATE
from birds.pann import read_audio_fast, get_model_predictions_for_clip, BIRDS, load_pretrained_model

_ = main
EXAMPLE_BIRDS = [
    'amered', 'comrav', 'dowwoo', 'eucdov', 'herthr', 'norcar', 'redcro',
    'normoc', 'pinsis', 'rebnut', 'rebwoo', 'sonspa', 'warvir', 'whbnut', 'whcspa',
]
choices = [ui.choice(ebird, tx.loc[ebird].comName) for ebird in EXAMPLE_BIRDS]

explore_cards = ['ebird_info', 'observation_season', 'observation_map', 'mel0', 'mel1', 'mel2']
recognize_cards = ['uploaded_audio_mel', 'top_bird_bar', 'top_bird_line', 'link']
reference_cards = ['references']


@app('/')
async def stg(q: Q):
    print('=' * 10)
    print(q.args)
    await display_main_page(q)
    await q.page.save()


async def del_cards(q: Q, cards):
    for c in cards:
        del q.page[c]
    await q.page.save()


async def add_progress(q: Q):
    q.page['progress'] = ui.form_card(
        box='4 2 5 2',
        items=[ui.progress(label='Chirp Chirp!')]
    )
    await q.page.save()


async def show_bird_dashboard(q: Q):
    ebird_code = q.client.ebird_code
    print(ebird_code)
    q.page['ebird_info'] = ui.form_card(
        box='1 4 3 6',
        items=[
            ui.text(await create_bird_info(q, ebird_code)),
            ui.link('More info from ebird.org', f'https://ebird.org/species/{ebird_code}')
        ]
    )

    season = pd.read_csv(f'./data/season/month_{ebird_code}.csv')
    rows = [(m, c) for m, c in season[['month', 'cnt']].values]
    q.page['observation_season'] = ui.plot_card(
        box='4 2 5 3',
        title='Observations across North America in the last two years',
        data=data('month observations', season.shape[0], rows),
        plot=ui.plot([ui.mark(type='area', x_scale='time', x='=month', y='=observations', y_min=0)])
    )

    q.page['observation_map'] = ui.frame_card(
        box='4 5 5 5',
        title='',
        content=geo_plot(ebird_code, 5, 5)
    )

    for i, y in enumerate(range(1, 10, 3)):
        q.page[f'mel{i}'] = ui.image_card(
            box=f'9 {y} 3 3',
            title='',
            type='png',
            image=create_random_spectogram(ebird_code))


async def show_recognize_dashboard(q: Q, audio_path, uploaded):
    filename = audio_path.split('/')[-1]
    y = read_audio_fast(audio_path)
    q.page['uploaded_audio_mel'] = ui.image_card(
        box='4 2 8 5',
        title='',
        type='png',
        image=create_spectogram_from_audio(y, filename, fig_size=(11.5, 4.2))
    )

    predictions = get_model_predictions_for_clip(y, q.app.model)
    class_probs = predictions[BIRDS].sum().reset_index()
    class_probs.columns = ['ebird', 'p']
    class_probs = class_probs.sort_values(by='p')

    top_birds = class_probs.tail(5).copy()
    top_birds['color'] = ['#CECECE', '#CECECE', '#3D3D3D', '#000000', '#FEE200']
    top_birds['dash'] = ['dot', 'dot', 'dash', 'dash', 'solid']
    top_birds['fill'] = [None, None, None, None, 'tozeroy']

    q.page['top_bird_bar'] = ui.frame_card(
        box='1 7 3 5',
        title='',
        content=top_bird_bar_plot(top_birds, 3, 5)
    )

    q.page['top_bird_line'] = ui.frame_card(
        box='4 7 8 5',
        title='',
        content=top_bird_line_chart(predictions, top_birds, 8, 5)
    )

    top_ebird = class_probs.ebird.values[-1]
    q.page['link'] = ui.form_card(
        box='1 2 3 5',
        items=[
            ui.markup(AUDIO_TEMPLATE.format(uploaded=uploaded)),
            ui.text(f'Best guess: [{tx.loc[top_ebird].comName}](https://ebird.org/species/{top_ebird})', size='l'),
            ui.button(name='new_recording', label='New recording', primary=True),
        ]
    )


async def display_main_page(q):
    if not q.app.initialized:
        q.app.initialized = True
        q.app.model = load_pretrained_model()
    if not q.client.initialized:
        q.client.initialized = True
        q.client.ebird_code = 'amered'
        q.client.current_hash = 'explore'
        q.page['tabs'] = ui.tab_card(
            box='4 1 5 1',
            items=[
                ui.tab(name='#explore', label='Explore'),
                ui.tab(name='#recognize', label='Recognize'),
                ui.tab(name='#references', label='References'),
            ]
        )
        q.page['header'] = ui.header_card(
            box='1 1 3 1',
            title='Bird song analysis',
            subtitle='And now for something completely different!',
            icon='MusicInCollectionFill',
            icon_color='yellow',
        )

    if q.args['#']:
        print(q.client.current_hash, '->', q.args['#'])
        q.client.current_hash = q.args['#']

    if q.args.ebird_code:
        q.client.ebird_code = q.args.ebird_code

    if q.client.current_hash == 'explore':
        await del_cards(q, recognize_cards + ['upload', 'progress', 'references'])
        q.page['bird_selector'] = ui.form_card(
            box='1 2 3 2', items=[
                ui.dropdown(name='ebird_code', label='Pick a bird', value=q.client.ebird_code, required=True,
                            choices=choices, trigger=True),
                ui.button(name='show_inputs', label='Show bird', primary=True),
            ]
        )
        if q.args.show_inputs:
            await del_cards(q, explore_cards)
            await add_progress(q)
            await show_bird_dashboard(q)

    if q.client.current_hash == 'recognize':
        await del_cards(q, explore_cards + recognize_cards + ['bird_selector', 'progress', 'references'])
        q.page['upload'] = ui.form_card(
            box='1 2 3 5', items=[
                ui.file_upload(name='audio_upload', label='Recognize!', multiple=False, file_extensions=['mp3', 'wav']),
                ui.button(name='example_audio', label='Use example audio', primary=True),
            ])

        if q.args.example_audio:
            await add_progress(q)
            audio_path = './data/audio/norcar_youtube.mp3'
            uploaded, = await q.site.upload([audio_path])
            await show_recognize_dashboard(q, audio_path, uploaded)

        if q.args.audio_upload:
            await add_progress(q)
            uploaded = q.args.audio_upload[0]
            audio_path = await q.site.download(uploaded, './temp/')
            await show_recognize_dashboard(q, audio_path, uploaded)

    if q.args.new_recording:
        await del_cards(q, recognize_cards)

    if q.client.current_hash == 'references':
        await del_cards(q, explore_cards + recognize_cards + ['upload', 'bird_selector', 'progress'])
        q.page['references'] = ui.form_card(
            box='1 2 8 5',
            items=[ui.text(show_references())]
        )
