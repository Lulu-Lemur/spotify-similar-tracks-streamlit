import streamlit as st
import spotipy
import numpy as np
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="SimilarTracks", page_icon=":musical_note:")
st.title("類似した曲を他のアーティストから探す")

# ----------------------
# Spotipy 認証
# ----------------------
if "SPOTIFY_CLIENT_ID" not in st.secrets or "SPOTIFY_CLIENT_SECRET" not in st.secrets:
    st.error("Streamlit Secrets に SPOTIFY_CLIENT_ID と SPOTIFY_CLIENT_SECRET を設定してください。")
    st.stop()

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=st.secrets["SPOTIFY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
    ))
except spotipy.SpotifyException as e:
    st.error(f"Spotify 認証エラー: {e}")
    st.stop()

# ----------------------
# ユーティリティ関数
# ----------------------
def extract_id(s):
    """Spotify URL/URI から ID を抽出"""
    if not s: return None
    s = s.strip().rstrip("/")
    if "open.spotify.com" in s:
        parts = s.split("/")
        if len(parts) >= 5:
            return parts[4].split("?")[0]
    if s.startswith("spotify:"):
        return s.split(":")[-1]
    return None

def get_track_id(q):
    """曲名や URL から track ID を取得"""
    id_ = extract_id(q)
    if id_: return id_
    try:
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        return items[0]["id"] if items else None
    except spotipy.SpotifyException as e:
        st.warning(f"Track search failed: {q}, {e}")
        return None

def get_artist_id(q):
    """アーティスト名や URL から artist ID を取得"""
    id_ = extract_id(q)
    if id_: return id_
    try:
        res = sp.search(q=q, type="artist", limit=1)
        items = res.get("artists", {}).get("items", [])
        return items[0]["id"] if items else None
    except spotipy.SpotifyException as e:
        st.warning(f"Artist search failed: {q}, {e}")
        return None

def get_artist_track_ids(artist_id, limit=300):
    """アーティストの曲を取得（ID, 曲名, アーティスト名, アルバム名）"""
    ids = []
    seen = set()
    offset = 0
    while True:
        try:
            res = sp.artist_albums(artist_id, album_type="album,single", limit=50, offset=offset)
            items = res.get("items", [])
            if not items: break
            for album in items:
                tracks = sp.album_tracks(album["id"]).get("items", [])
                for t in tracks:
                    if t["id"] not in seen:
                        ids.append((t["id"], t["name"], ", ".join([ar["name"] for ar in t["artists"]]), album["name"]))
                        seen.add(t["id"])
                        if len(ids) >= limit: return ids
            offset += len(items)
        except spotipy.SpotifyException as e:
            st.warning(f"Artist tracks fetch failed: {e}")
            break
    return ids

def audio_features_for_ids(ids):
    """曲 ID リストから音響特徴をまとめて取得"""
    feats = []
    for i in range(0, len(ids), 20):
        batch = [x for x in ids[i:i+20] if x]
        if not batch: continue
        try:
            res = sp.audio_features(batch)
            feats.extend([f if f else {"id": batch[j], "danceability":np.nan,"energy":np.nan,"valence":np.nan,"tempo":np.nan,
                                       "acousticness":np.nan,"instrumentalness":np.nan,"liveness":np.nan} for j,f in enumerate(res)])
        except spotipy.SpotifyException as e:
            st.warning(f"Audio features fetch failed for batch {batch[:5]}...: {e}")
            feats.extend([{"id":x,"danceability":np.nan,"energy":np.nan,"valence":np.nan,"tempo":np.nan,
                           "acousticness":np.nan,"instrumentalness":np.nan,"liveness":np.nan} for x in batch])
    return feats

# ----------------------
# Streamlit UI
# ----------------------
track_a = st.text_input("Input your favorite music A（曲名 or Spotify URL）")
track_b = st.text_input("Input your favorite music B（曲名 or Spotify URL）")
artist_bb = st.text_input("Input your favorite artist（名前 or Spotify URL）")
topk = st.number_input("返す件数", min_value=1, max_value=50, value=10)

if st.button("おすすめを探す"):
    with st.spinner("検索中..."):
        ida = get_track_id(track_a)
        idb = get_track_id(track_b)
        if not ida or not idb:
            st.error("曲が見つかりません。入力を確認してください。")
            st.stop()

        id_artist = get_artist_id(artist_bb)
        if not id_artist:
            st.error("アーティストが見つかりません。")
            st.stop()

        ids_meta = get_artist_track_ids(id_artist, limit=300)
        if not ids_meta:
            st.error("アーティストの曲が見つかりません。")
            st.stop()

        # ----------------------
        # 曲メタ情報取得
        # ----------------------
        ids = [i[0] for i in ids_meta]
        meta_batches = []
        for i in range(0, len(ids), 50):
            try:
                meta_batches.extend(sp.tracks(ids[i:i+50])["tracks"])
            except spotipy.SpotifyException as e:
                st.warning(f"Track metadata fetch failed for batch {ids[i:i+5]}...: {e}")

        # ----------------------
        # 音響特徴取得
        # ----------------------
        track_ids = [t[0] for t in ids_meta if t[0]]
        feats = audio_features_for_ids(track_ids)

        # DataFrame 作成
        df = pd.DataFrame([{"id": m["id"], "name": m["name"],
                            "artists": ", ".join([a["name"] for a in m["artists"]]),
                            "album": m["album"]["name"]} for m in meta_batches])
        feat_keys = ['danceability','energy','valence','tempo','acousticness','instrumentalness','liveness']
        feat_rows = [{k: f[k] for k in feat_keys} for f in feats if f]
        df_feat = pd.DataFrame(feat_rows)
        df = pd.concat([df.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1).dropna()

        # ターゲット曲の特徴
        t_feats = audio_features_for_ids([ida, idb])
        t_feats = [f for f in t_feats if f]  # None を除外
        if not t_feats:
            st.error("入力曲の音響特徴が取得できません")
            st.stop()

        # ----------------------
        # 類似度計算
        # ----------------------
        combined = pd.concat([df[feat_keys], pd.DataFrame(t_feats)[feat_keys]], ignore_index=True)
        mn, mx = combined.min(), combined.max()
        df_norm = (df[feat_keys] - mn) / (mx - mn + 1e-9)
        t_norm = (pd.DataFrame(t_feats)[feat_keys] - mn) / (mx - mn + 1e-9)
        target_vec = t_norm.mean().values

        def cos_sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        df["sim"] = [cos_sim(target_vec, row.values) for _, row in df_norm.iterrows()]
        df = df.sort_values("sim", ascending=False).head(int(topk))

        # ----------------------
        # 結果表示
        # ----------------------
        for _, r in df.iterrows():
            st.markdown(f"**{r['name']}** — {r['artists']} — _{r['album']}_ (類似度: {r['sim']:.3f})")
            st.markdown(f"https://open.spotify.com/track/{r['id']}")
            preview = r.get("preview_url")
            if preview: st.audio(preview)
