"""Browse the locally downloaded, publisher-labeled intersection image subset."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1] / 'data/intersection_anomalies'
st.set_page_config(page_title='Intersection images', page_icon='🚦', layout='wide')
st.title('Intersection image dataset')
st.caption('Real aerial photographs · UIT-ADrone · normal / anomaly classification')

if not (ROOT / 'labels.csv').exists():
    st.info('The image dataset has not been prepared yet.')
    st.stop()

df = pd.read_csv(ROOT / 'labels.csv', dtype={'sample_id': str})
available = df['image_path'].map(lambda p: (ROOT / p).is_file())
cols = st.columns(3)
cols[0].metric('Images downloaded', int(available.sum()), help='Target: 1,500 original aerial images')
cols[1].metric('Normal', int(((df.label == 0) & available).sum()))
cols[2].metric('Anomaly', int(((df.label == 1) & available).sum()))
st.info('Labels come from the dataset authors. “Anomaly” means a source-annotated event, '
        'not a verified legal violation. Specific event types and responsible objects '
        'have not been independently annotated in this subset.')

left, middle, right = st.columns(3)
partition = left.selectbox('Dataset split', ['All', 'train', 'validation', 'test'])
label = middle.selectbox('Scene label', ['All', 'normal', 'anomaly'])
recording = right.selectbox('Recording', ['All'] + sorted(df.source_video.unique()))
filtered = df[available]
if partition != 'All':
    filtered = filtered[filtered.split == partition]
if label != 'All':
    filtered = filtered[filtered.class_name == label]
if recording != 'All':
    filtered = filtered[filtered.source_video == recording]
filtered = filtered.reset_index(drop=True)

if filtered.empty:
    st.warning('No downloaded images match these filters yet.')
else:
    selection_key = f'image_{partition}_{label}_{recording}'
    number = st.number_input('Image number', min_value=1, max_value=len(filtered), value=1, key=selection_key)
    row = filtered.iloc[number-1]
    st.image(str(ROOT / row.image_path), caption=f'{row.class_name.upper()} · {row.split} · {row.source_video} · {row.source_filename}', width='stretch')
    st.caption(f'Image {number} of {len(filtered)} matching images. Sample ID: {row.sample_id}')
    st.link_button('View original source', row.source_url)
    with st.expander('Label and source details'):
        st.json(row.fillna('').to_dict())

st.download_button('Download all labels (CSV)', (ROOT / 'labels.csv').read_bytes(), 'intersection_labels.csv', 'text/csv')
if (ROOT / 'verification.json').exists():
    with st.expander('Dataset verification'):
        st.json(json.loads((ROOT / 'verification.json').read_text()))
st.markdown('[Dataset source and citation](https://uit-together.github.io/datasets/UIT-ADrone/)')
st.caption('The 80/20 ratio describes scene classes. Training, validation and test sets '
           'use a separate 70/15/15 split by recording. Locations can overlap. '
           'The traffic-count LSTM has not been trained on these images.')
