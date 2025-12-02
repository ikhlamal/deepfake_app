import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import mediapipe as mp

# ==== CSS custom jarak, centering, dan border video ====
st.markdown("""
    <style>
    .block-container {padding-top: 2rem !important; padding-bottom: 0.5rem !important;}
    .stPlotlyChart {padding-top: 0rem !important; padding-bottom: 0rem !important; margin-top: 5px !important; margin-bottom: 0px !important;}
    .element-container {padding-top: 0rem !important; padding-bottom: 0rem !important;}
    .st-emotion-cache-1wmyrj9, .st-emotion-cache-1r6slb0 {padding-top: 0.2rem !important; padding-bottom: 0.2rem !important;}
    /* Video centering + border */
    .element-container:has(video) {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    video {
        border: 3px solid #888;
        border-radius: 14px;
        box-shadow:0 2px 8px 0 #8884;
        background: #fff;
        margin: 12px auto 18px auto;
        display:block;
        max-width: 320px !important;
        height: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="LipSync Deepfake Video Detection", layout="wide")
st.title("LipSync Deepfake Video Detection")

FRAME_COUNT = 8
RESIDUE_COUNT = 7
FRAME_SHAPE = (64, 144)

# -------------------------
# v1
# -------------------------
def build_lipinc_model_v1(frame_shape=(8,64,144,3), residue_shape=(7,64,144,3), d_model=128):

    from tensorflow.keras.layers import Input, Lambda, Dense, Layer, GlobalAveragePooling1D, LayerNormalization
    from tensorflow.keras.models import Model

    class VisionTemporalTransformer(Layer):
        def __init__(self, patch_size=8, d_model=128, num_heads=4, spatial_layers=1, temporal_layers=1, **kwargs):
            super(VisionTemporalTransformer, self).__init__(**kwargs)
            self.patch_size = patch_size
            self.d_model = d_model
            self.num_heads = num_heads
            self.spatial_layers = spatial_layers
            self.temporal_layers = temporal_layers
            self.dense_projection = Dense(d_model)
            self.pos_emb = None
            self.spatial_mhas = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(spatial_layers)]
            self.spatial_norm1 = [LayerNormalization() for _ in range(spatial_layers)]
            self.spatial_ffn = [tf.keras.Sequential([Dense(d_model*4, activation='relu'), Dense(d_model)]) for _ in range(spatial_layers)]
            self.spatial_norm2 = [LayerNormalization() for _ in range(spatial_layers)]
            self.temporal_mhas = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(temporal_layers)]
            self.temporal_norm1 = [LayerNormalization() for _ in range(temporal_layers)]
            self.temporal_ffn = [tf.keras.Sequential([Dense(d_model*4, activation='relu'), Dense(d_model)]) for _ in range(temporal_layers)]
            self.temporal_norm2 = [LayerNormalization() for _ in range(temporal_layers)]
        def build(self, input_shape):
            _, frames, H, W, C = input_shape
            ph = H // self.patch_size
            pw = W // self.patch_size
            num_patches = ph * pw
            self.pos_emb = self.add_weight(shape=(1, num_patches, self.d_model), initializer='random_normal', trainable=True, name='pos_emb')
            super(VisionTemporalTransformer, self).build(input_shape)
        def call(self, inputs):
            batch = tf.shape(inputs)[0]
            frames = tf.shape(inputs)[1]
            H = tf.shape(inputs)[2]
            W = tf.shape(inputs)[3]
            C = tf.shape(inputs)[4]
            reshaped = tf.reshape(inputs, (-1, H, W, C))
            patches = tf.image.extract_patches(
                images=reshaped,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1,1,1,1],
                padding='VALID'
            )
            num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
            patch_dim = tf.shape(patches)[-1]
            patches = tf.reshape(patches, (-1, num_patches, patch_dim))
            x = self.dense_projection(patches) + self.pos_emb
            for i in range(self.spatial_layers):
                attn = self.spatial_mhas[i](query=x, value=x, key=x)
                x = self.spatial_norm1[i](x + attn)
                ff = self.spatial_ffn[i](x)
                x = self.spatial_norm2[i](x + ff)
            d_model = tf.shape(x)[-1]
            x = tf.reshape(x, (batch, frames, -1, d_model))
            x = tf.reduce_mean(x, axis=2)
            for i in range(self.temporal_layers):
                attn = self.temporal_mhas[i](query=x, value=x, key=x)
                x = self.temporal_norm1[i](x + attn)
                ff = self.temporal_ffn[i](x)
                x = self.temporal_norm2[i](x + ff)
            pooled = GlobalAveragePooling1D()(x)
            return pooled
        def get_config(self):
            config = super().get_config()
            config.update({
                "patch_size": self.patch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "spatial_layers": self.spatial_layers,
                "temporal_layers": self.temporal_layers,
            })
            return config

    frame_input = Input(shape=frame_shape, name='FrameInput')
    residue_input = Input(shape=residue_shape, name='ResidueInput')

    vt = VisionTemporalTransformer(
        patch_size=8,
        d_model=d_model,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1
    )

    frame_feat = vt(frame_input)
    residue_feat = vt(residue_input)

    expand1 = Lambda(lambda x: tf.expand_dims(x, axis=1))
    q = expand1(frame_feat)
    k = expand1(residue_feat)
    v = k

    mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model//4)
    attn_out = mha(query=q, value=v, key=k)
    squeeze = Lambda(lambda x: tf.squeeze(x, axis=1))
    attn_out = squeeze(attn_out)

    concat = Lambda(lambda t: tf.concat(t, axis=1))
    fusion = concat([frame_feat, residue_feat, attn_out])

    x = Dense(512, activation='relu')(fusion)
    x = Dense(256, activation='relu')(x)

    class_output = Dense(2, activation='softmax', name='class_output')(x)
    features_output = Dense(d_model, activation=None, name='features_output')(x)

    model = Model(
        inputs=[frame_input, residue_input],
        outputs=[class_output, features_output],
        name='LIPINC_v1'
    )
    return model

# ---------- v2, v3, v4, v5: Custom MHA ----------
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = num_heads * key_dim
        self.query_dense = tf.keras.layers.Dense(self.d_model)
        self.key_dense = tf.keras.layers.Dense(self.d_model)
        self.value_dense = tf.keras.layers.Dense(self.d_model)
        self.combine_heads = tf.keras.layers.Dense(self.d_model)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, query, value, key):
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.combine_heads(output)
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

class VisionTemporalTransformer(tf.keras.layers.Layer):
    def __init__(self, patch_size=8, d_model=128, num_heads=4, spatial_layers=1, temporal_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        self.dense_projection = tf.keras.layers.Dense(d_model)
        self.spatial_mhas = [MultiHeadAttention(num_heads, d_model//num_heads) for _ in range(spatial_layers)]
        self.spatial_norm1 = [tf.keras.layers.LayerNormalization() for _ in range(spatial_layers)]
        self.spatial_ffn = [tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ]) for _ in range(spatial_layers)]
        self.spatial_norm2 = [tf.keras.layers.LayerNormalization() for _ in range(spatial_layers)]
        self.temporal_mhas = [MultiHeadAttention(num_heads, d_model//num_heads) for _ in range(temporal_layers)]
        self.temporal_norm1 = [tf.keras.layers.LayerNormalization() for _ in range(temporal_layers)]
        self.temporal_ffn = [tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ]) for _ in range(temporal_layers)]
        self.temporal_norm2 = [tf.keras.layers.LayerNormalization() for _ in range(temporal_layers)]
    def build(self, input_shape):
        H = input_shape[2]
        W = input_shape[3]
        ph = H // self.patch_size
        pw = W // self.patch_size
        num_patches = ph * pw
        self.pos_emb = self.add_weight(shape=(1, num_patches, self.d_model), initializer='random_normal', trainable=True, name='pos_emb')
        super().build(input_shape)
    def call(self, inputs):
        input_shape = inputs.get_shape()
        shape = tf.shape(inputs)
        batch = shape[0]
        frames = shape[1]
        H = shape[2]
        W = shape[3]
        C_static = input_shape[-1]
        C = C_static if C_static is not None else shape[4]
        reshaped = tf.reshape(inputs, (-1, H, W, C))
        patches = tf.image.extract_patches(
            images=reshaped,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        patch_dim_static = self.patch_size * self.patch_size * (C_static or 3)
        patches = tf.reshape(patches, (-1, tf.shape(patches)[1] * tf.shape(patches)[2], patch_dim_static))
        x = self.dense_projection(patches) + self.pos_emb
        for i in range(self.spatial_layers):
            attn = self.spatial_mhas[i](x, value=x, key=x)
            x = self.spatial_norm1[i](x + attn)
            ff = self.spatial_ffn[i](x)
            x = self.spatial_norm2[i](x + ff)
        x = tf.reshape(x, (batch, frames, -1, self.d_model))
        x = tf.reduce_mean(x, axis=2)
        x.set_shape([None, None, self.d_model])
        for i in range(self.temporal_layers):
            attn = self.temporal_mhas[i](x, value=x, key=x)
            x = self.temporal_norm1[i](x + attn)
            ff = self.temporal_ffn[i](x)
            x = self.temporal_norm2[i](x + ff)
        pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
        return pooled
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "spatial_layers": self.spatial_layers,
            "temporal_layers": self.temporal_layers,
        })
        return config

def build_lipinc_model(frame_shape=(8,64,144,3), residue_shape=(7,64,144,3), d_model=128):
    from tensorflow.keras.layers import Input, Lambda, Dense
    from tensorflow.keras.models import Model
    frame_input = Input(shape=frame_shape, name='FrameInput')
    residue_input = Input(shape=residue_shape, name='ResidueInput')
    vt = VisionTemporalTransformer(patch_size=8, d_model=d_model, num_heads=4, spatial_layers=1, temporal_layers=1)
    frame_feat = vt(frame_input)
    residue_feat = vt(residue_input)
    expand1 = Lambda(lambda x: tf.expand_dims(x, axis=1))
    q = expand1(frame_feat)
    k = expand1(residue_feat)
    v = k
    mha = MultiHeadAttention(num_heads=4, key_dim=d_model//4)
    attn_out = mha(q, value=v, key=k)
    squeeze = Lambda(lambda x: tf.squeeze(x, axis=1))
    attn_out = squeeze(attn_out)
    concat = Lambda(lambda t: tf.concat(t, axis=1))
    fusion = concat([frame_feat, residue_feat, attn_out])
    x = Dense(512, activation='relu')(fusion)
    x = Dense(256, activation='relu')(x)
    class_output = Dense(2, activation='softmax', name='class_output')(x)
    features_output = Dense(d_model, activation=None, name='features_output')(x)
    model = Model(inputs=[frame_input, residue_input], outputs=[class_output, features_output], name='LIPINC_fixed')
    return model

# ----- Crop mulut khusus v5 -----
class MouthExtractor:
    MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    def crop(self, img):
        result = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return cv2.resize(img, (144, 64))
        h, w, _ = img.shape
        pts = np.array([[int(lm.x*w), int(lm.y*h)] for lm in result.multi_face_landmarks[0].landmark])
        try:
            mouth_pts = pts[self.MOUTH_LANDMARKS]
        except:
            return cv2.resize(img, (144, 64))
        x, y, w_m, h_m = cv2.boundingRect(mouth_pts)
        pad_w, pad_h = int(w_m*0.2), int(h_m*0.5)
        x1, y1 = max(0, x-pad_w), max(0, y-pad_h)
        x2, y2 = min(w, x+w_m+pad_w), min(h, y+h_m+pad_h)
        crop = img[y1:y2, x1:x2]
        crop_resize = cv2.resize(crop, (144, 64))
        return crop_resize

def load_video_frames(file_bytes, frame_count=FRAME_COUNT, dim=FRAME_SHAPE):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    try:
        while len(frames) < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (dim[1], dim[0]))
            frames.append(frame)
    finally:
        cap.release()
    import os
    os.remove(tmp_path)
    frames = np.array(frames)
    if len(frames) == 0:
        return np.zeros((frame_count, dim[0], dim[1], 3), dtype=np.float32)
    if len(frames) < frame_count:
        padding = np.zeros((frame_count - len(frames), dim[0], dim[1], 3), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)
    return frames.astype(np.float32) / 255.0

def load_video_frames_v5(file_bytes, frame_count=FRAME_COUNT):
    import tempfile
    mouth_extractor = MouthExtractor()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    try:
        while len(frames) < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            crop = mouth_extractor.crop(frame)
            frames.append(crop)
    finally:
        cap.release()
    import os
    os.remove(tmp_path)
    frames = np.array(frames)
    if len(frames) == 0:
        return np.zeros((frame_count, 64, 144, 3), dtype=np.float32)
    if len(frames) < frame_count:
        padding = np.zeros((frame_count - len(frames), 64, 144, 3), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)
    return frames.astype(np.float32) / 255.0

def compute_residue(frames):
    residues = np.zeros((RESIDUE_COUNT, 64, 144, 3), dtype=np.float32)
    for i in range(1, len(frames)):
        residues[i-1] = frames[i] - frames[i-1]
    return residues

@st.cache_resource(show_spinner=True)
def load_model_and_weights_v1(weight_path):
    model = build_lipinc_model_v1()
    model.load_weights(weight_path)
    return model

@st.cache_resource(show_spinner=True)
def load_model_and_weights(weight_path):
    model = build_lipinc_model()
    model.load_weights(weight_path)
    return model

model_v1 = load_model_and_weights_v1("v1.h5")
model_v2 = load_model_and_weights("v2.h5")
model_v3 = load_model_and_weights("v3.h5")
model_v4 = load_model_and_weights("v4.h5")
model_v5 = load_model_and_weights("v5.h5")

def gauge_chart(score, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score*100,
        gauge={
            'axis': {'range': [0,100], 'tickwidth':1},
            'bar': {'color': 'red' if score > 0.5 else 'green'},
            'steps': [
                {'range': [0,50], 'color': 'lightgreen'},
                {'range': [50,100], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': score*100
            }
        },
        number={'suffix': "%", 'font': {'size': 33}},
        title={'text': ""}
    ))
    fig.update_layout(margin=dict(t=12, b=12, l=8, r=8), height=210)
    return fig

def center_text(text, color="inherit", font_size="19px", weight="600", margin="12px 0 8px 0"):
    return f"""
    <div style='width:100%;display:flex;justify-content:center;align-items:center; margin:{margin};'>
      <span style='color:{color};font-size:{font_size};font-weight:{weight};'>{text}</span>
    </div>"""

uploaded = st.file_uploader("Upload video mp4", type=["mp4"])
if uploaded is not None:
    st.video(uploaded, start_time=0)
    file_bytes = uploaded.read()
    frames_std = load_video_frames(file_bytes)
    residues_std = compute_residue(frames_std)
    X_frames_std = np.expand_dims(frames_std, axis=0)
    X_residues_std = np.expand_dims(residues_std, axis=0)

    # --- Prediksi ---
    pred_class_v1, _ = model_v1.predict([X_frames_std, X_residues_std])
    score_v1 = float(pred_class_v1[0][1])
    label_v1 = "FAKE" if score_v1 > 0.5 else "REAL"

    pred_class_v2, _ = model_v2.predict([X_frames_std, X_residues_std])
    score_v2 = float(pred_class_v2[0][1])
    label_v2 = "FAKE" if score_v2 > 0.5 else "REAL"

    pred_class_v3, _ = model_v3.predict([X_frames_std, X_residues_std])
    score_v3 = float(pred_class_v3[0][1])
    label_v3 = "FAKE" if score_v3 > 0.5 else "REAL"

    pred_class_v4, _ = model_v4.predict([X_frames_std, X_residues_std])
    score_v4 = float(pred_class_v4[0][1])
    label_v4 = "FAKE" if score_v4 > 0.5 else "REAL"

    # --- v5: pipeline crop mouth ---
    frames_v5 = load_video_frames_v5(file_bytes)
    residues_v5 = compute_residue(frames_v5)
    X_frames_v5 = np.expand_dims(frames_v5, axis=0)
    X_residues_v5 = np.expand_dims(residues_v5, axis=0)
    pred_class_v5, _ = model_v5.predict([X_frames_v5, X_residues_v5])
    score_v5 = float(pred_class_v5[0][1])
    label_v5 = "FAKE" if score_v5 > 0.5 else "REAL"

    # ---- Layout 5 kolom (unique key plotly_chart) ----
    colv1, colv2, colv3, colv4, colv5 = st.columns(5)
    with colv1:
        with st.container(border=True):
            st.markdown(center_text("Model V1", font_size="20px", weight="700", margin="10px 0 0px 0"), unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(score_v1, label_v1), use_container_width=True, key="plot_v1")
            st.markdown(center_text(label_v1, color="red" if label_v1=="FAKE" else "green", font_size="18px", weight="700", margin="10px 0 12px 0"), unsafe_allow_html=True)
    with colv2:
        with st.container(border=True):
            st.markdown(center_text("Model V2", font_size="20px", weight="700", margin="10px 0 0px 0"), unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(score_v2, label_v2), use_container_width=True, key="plot_v2")
            st.markdown(center_text(label_v2, color="red" if label_v2=="FAKE" else "green", font_size="18px", weight="700", margin="10px 0 12px 0"), unsafe_allow_html=True)
    with colv3:
        with st.container(border=True):
            st.markdown(center_text("Model V3", font_size="20px", weight="700", margin="10px 0 0px 0"), unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(score_v3, label_v3), use_container_width=True, key="plot_v3")
            st.markdown(center_text(label_v3, color="red" if label_v3=="FAKE" else "green", font_size="18px", weight="700", margin="10px 0 12px 0"), unsafe_allow_html=True)
    with colv4:
        with st.container(border=True):
            st.markdown(center_text("Model V4", font_size="20px", weight="700", margin="10px 0 0px 0"), unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(score_v4, label_v4), use_container_width=True, key="plot_v4")
            st.markdown(center_text(label_v4, color="red" if label_v4=="FAKE" else "green", font_size="18px", weight="700", margin="10px 0 12px 0"), unsafe_allow_html=True)
    with colv5:
        with st.container(border=True):
            st.markdown(center_text("Model V5", font_size="20px", weight="700", margin="10px 0 0px 0"), unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(score_v5, label_v5), use_container_width=True, key="plot_v5")
            st.markdown(center_text(label_v5, color="red" if label_v5=="FAKE" else "green", font_size="18px", weight="700", margin="10px 0 12px 0"), unsafe_allow_html=True)
