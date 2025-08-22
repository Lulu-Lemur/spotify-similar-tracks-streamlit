import streamlit as st,spotipy,numpy as np,pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="SimilarTracks",page_icon=":musical_note:")
st.title("類似した曲を他のアーティストから探す")

if "SPOTIFY_CLIENT_ID" not in st.secrets or "SPOTIFY_CLIENT_SECRET" not in st.secrets:
    st.error("Deploy前に Streamlit Secrets に SPOTIFY_CLIENT_ID と SPOTIFY_CLIENT_SECRET を設定してください。"); st.stop()

sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=st.secrets["SPOTIFY_CLIENT_ID"],client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]))

def extract_id(s):
    if not s: return None
    s=s.strip()
    if "open.spotify.com" in s: return s.split("/")[-1].split("?")[0]
    if s.startswith("spotify:"): return s.split(":")[-1]
    return None

def get_track_id(q):
    id=extract_id(q)
    if id: return id
    r=sp.search(q=q,type="track",limit=1)
    items=r.get("tracks",{}).get("items",[])
    return items[0]["id"] if items else None

def get_artist_id(q):
    id=extract_id(q)
    if id: return id
    r=sp.search(q=q,type="artist",limit=1)
    items=r.get("artists",{}).get("items",[])
    return items[0]["id"] if items else None

def get_artist_track_ids(artist_id,limit=300):
    ids=[];seen=set();offset=0
    while True:
        res=sp.artist_albums(artist_id,album_type="album,single",limit=50,offset=offset)
        items=res.get("items",[])
        if not items: break
        for a in items:
            tr=sp.album_tracks(a["id"])
            for t in tr.get("items",[]):
                if t["id"] not in seen:
                    ids.append((t["id"],t["name"],", ".join([ar["name"] for ar in t["artists"]]),a["name"]))
                    seen.add(t["id"]); 
                    if len(ids)>=limit: return ids
        offset+=len(items)
    return ids

def audio_features_for_ids(ids):
    feats=[]
    for i in range(0,len(ids),50):
        batch = [x for x in ids[i:i+50] if x]  # None除去
        if batch:
            feats.extend(sp.audio_features(batch))
    return feats

track_a=st.text_input("Input your favorit music A（曲名 or Spotify URL）")
track_b=st.text_input("Input your favorit music B（曲名 or Spotify URL）")
artist_bb=st.text_input("Input your favorit artist（名前 or Spotify URL）")
topk=st.number_input("返す件数",min_value=1,max_value=50,value=10)

if st.button("おすすめを探す"):
    with st.spinner("検索中..."):
        ida=get_track_id(track_a); idb=get_track_id(track_b)
        if not ida or not idb:
            st.error("AAA/BBB の曲が見つかりません。入力を確認してください。"); st.stop()
        id_artist=get_artist_id(artist_bb)
        if not id_artist:
            st.error("BB のアーティストが見つかりません。"); st.stop()
        ids_meta=get_artist_track_ids(id_artist,limit=300)
        if not ids_meta:
            st.error("BB の曲が見つかりません。"); st.stop()
        ids=[i[0] for i in ids_meta]
        meta_batches=[]
        for i in range(0,len(ids),50):
            meta_batches.extend(sp.tracks(ids[i:i+50])["tracks"])
        # feats=audio_features_for_ids(ids_meta)
        track_ids = [t[0] for t in ids_meta if t[0]]  # タプルからIDだけ抽出、None除去
        feats = audio_features_for_ids(track_ids)
        df=pd.DataFrame([{"id":m["id"],"name":m["name"],"artists":", ".join([a["name"] for a in m["artists"]]),"album":m["album"]["name"]} for m in meta_batches])
        feat_keys=['danceability','energy','valence','tempo','acousticness','instrumentalness','liveness']
        feat_rows=[{k:f[k] for k in feat_keys} for f in feats if f]
        df_feat=pd.DataFrame(feat_rows)
        df=pd.concat([df.reset_index(drop=True),df_feat.reset_index(drop=True)],axis=1).dropna()
        t_feats=audio_features_for_ids([ida,idb])
        if not t_feats or any(f is None for f in t_feats): st.error("AAA/BBB の音響特徴が取得できません"); st.stop()
        combined=pd.concat([df[feat_keys],pd.DataFrame(t_feats)[feat_keys]],ignore_index=True)
        mn=combined.min();mx=combined.max()
        df_norm=(df[feat_keys]-mn)/(mx-mn+1e-9)
        t_norm=(pd.DataFrame(t_feats)[feat_keys]-mn)/(mx-mn+1e-9)
        target_vec=t_norm.mean().values
        def cos_sim(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
        df["sim"]=[cos_sim(target_vec,row.values) for _,row in df_norm.iterrows()]
        df=df.sort_values("sim",ascending=False).head(int(topk))
        for _,r in df.iterrows():
            st.markdown(f"**{r['name']}** — {r['artists']}  — _{r['album']}_  (類似度: {r['sim']:.3f})")
            st.markdown(f"https://open.spotify.com/track/{r['id']}")
            preview=sp.track(r['id']).get("preview_url")
            if preview: st.audio(preview)
