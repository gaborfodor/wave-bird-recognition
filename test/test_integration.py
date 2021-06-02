from birds.display_utils import geo_plot
from birds.pann import load_pretrained_model, read_audio_fast, get_model_predictions_for_clip, BIRDS


def test_prediction_works():
    test_bird = 'comrav'

    model = load_pretrained_model()

    y = read_audio_fast(f'./data/audio/{test_bird}.mp3')

    predictions = get_model_predictions_for_clip(y, model)

    class_probs = predictions[BIRDS].sum().reset_index()
    class_probs.columns = ['ebird', 'p']
    class_probs = class_probs.sort_values(by='p')

    top_ebird = class_probs.ebird.values[-1]
    assert top_ebird == test_bird


def test_map():
    html = geo_plot('norcar', 10, 10)
    with open('./temp/test_map.html', 'w') as f:
        f.write(html)
