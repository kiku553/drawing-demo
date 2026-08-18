import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Streamlit app タイトル
st.set_page_config(page_title="図形描画デモ", layout="wide")
st.title("図形描画デモ")
st.caption("キャンバスに描いた手書き線をRDP法で簡略化し、「線分」「円弧」に判定してキャンバス上で清書します。")

CANVAS_W, CANVAS_H = 700, 450
LINE_COLOR = "#1f77b4"   # 線分の色
ARC_COLOR = "#2ca02c"    # 円弧の色


# ==========================================================
# 座標取得（fabric.js のパス → 点列）
# ==========================================================
def quad_bezier(p0, p1, p2, n=8):
    """2次ベジェ曲線を n 点にサンプリングする"""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = np.array(p0, float), np.array(p1, float), np.array(p2, float)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def path_to_points(path):
    """
    fabric.js のパス命令を (x, y) の点列に変換する。
    freedraw は ["M",x,y] / ["Q",cx,cy,x,y] / ["L",x,y] を返すので、
    命令ごとに引数の数が違う点に注意（Q の cmd[1],cmd[2] は制御点であって通過点ではない）。
    """
    pts, cur = [], None
    for cmd in path:
        op = cmd[0]
        if op in ("M", "L"):
            cur = [cmd[1], cmd[2]]
            pts.append(cur)
        elif op == "Q" and cur is not None:
            seg = quad_bezier(cur, [cmd[1], cmd[2]], [cmd[3], cmd[4]])
            pts.extend(seg[1:].tolist())
            cur = [cmd[3], cmd[4]]
        elif op == "C" and cur is not None:
            cur = [cmd[5], cmd[6]]
            pts.append(cur)

    if len(pts) < 2:
        return np.zeros((0, 2))

    pts = np.asarray(pts, dtype=float)
    # 連続する重複点を除去（重複があるとRDPでゼロ除算が起きる）
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1]) > 1e-9
    return pts[keep]


# ==========================================================
# RDP法アルゴリズム
# ==========================================================
def perpendicular_distance(point, start, end):
    """直線 start-end と点 point の距離を計算する"""
    x, y = point[0], point[1]
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]

    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        # start と end が同一点のときは点と点の距離を返す（ゼロ除算の回避）
        return float(np.hypot(x - x1, y - y1))

    return float(abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom)


def douglas_peucker(point_list, start, end, epsilon):
    """point_list[start:end] を簡略化した点列を返す（end は含まない）"""
    # 点が2個以下ならこれ以上簡略化できない（再帰の終了条件）
    if end - start <= 1:
        return [point_list[start]]
    if end - start == 2:
        return [point_list[start], point_list[end - 1]]

    # 直線からの距離が最大の点を見つける
    dmax, index = 0.0, start + 1
    for i in range(start + 1, end - 1):
        d = perpendicular_distance(point_list[i], point_list[start], point_list[end - 1])
        if d > dmax:
            index, dmax = i, d

    # 最大距離が閾値を超えた場合、再帰的に処理する
    if dmax > epsilon:
        rec_results1 = douglas_peucker(point_list, start, index + 1, epsilon)
        rec_results2 = douglas_peucker(point_list, index, end, epsilon)

        # 結果を結合する（分岐点が重複するので先頭を除いて連結）
        rec_results1.extend(rec_results2[1:])
        return rec_results1
    else:
        return [point_list[start], point_list[end - 1]]


def douglas_peucker_indices(points, epsilon):
    """RDPで残る点の「添字」を返す（線分/円弧の判定で元の点列を参照するため）"""
    n = len(points)
    if n < 3:
        return list(range(n))

    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:  # 再帰の深さ制限に引っかからないようスタックで実装
        s, e = stack.pop()
        if e - s < 2:
            continue
        dmax, index = 0.0, -1
        for i in range(s + 1, e):
            d = perpendicular_distance(points[i], points[s], points[e])
            if d > dmax:
                dmax, index = d, i
        if dmax > epsilon and index > 0:
            keep.add(index)
            stack.append((s, index))
            stack.append((index, e))
    return sorted(keep)


# ==========================================================
# 線分 / 円弧のあてはめ
# ==========================================================
def line_error(pts):
    """両端を結ぶ直線からの最大ずれ"""
    if len(pts) < 3:
        return 0.0
    return max(perpendicular_distance(p, pts[0], pts[-1]) for p in pts[1:-1])


def fit_circle(pts):
    """最小二乗による円あてはめ（Kasa法）。(cx, cy, r) を返す。失敗時は None"""
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    val = sol[2] + cx**2 + cy**2
    if not np.isfinite(val) or val <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(val))


def arc_error(pts, circle):
    """円からの最大ずれ"""
    cx, cy, r = circle
    return float(np.max(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)))


def arc_angles(pts, circle):
    """点列に沿って角度を連続化し、(開始角, 終了角) を返す"""
    cx, cy, _ = circle
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return float(ang[0]), float(ang[-1])

def make_line(points, i0, i1):
    sub = points[i0 : i1 + 1]
    return {
        "type": "線分", "i0": i0, "i1": i1,
        "start": sub[0], "end": sub[-1],
        "length": float(np.hypot(*(sub[-1] - sub[0]))),
        "error": line_error(sub),
    }


def make_arc(points, i0, i1, circle):
    sub = points[i0 : i1 + 1]
    a0, a1 = arc_angles(sub, circle)
    return {
        "type": "円弧", "i0": i0, "i1": i1,
        "center": (circle[0], circle[1]), "radius": circle[2],
        "a0": a0, "a1": a1,
        "error": arc_error(sub, circle),
        "start": sub[0], "end": sub[-1],
    }


def merge_primitives(points, prims, tol):
    """RDPの頂点が細かすぎて分断された図形を、同じ図形とみなせる範囲で結合する"""
    merged = []
    for p in prims:
        if merged:
            prev = merged[-1]
            sub = points[prev["i0"] : p["i1"] + 1]

            # 隣り合う線分どうしが1本の直線とみなせるなら結合
            if prev["type"] == "線分" and p["type"] == "線分" and line_error(sub) <= tol * 1.5:
                merged[-1] = make_line(points, prev["i0"], p["i1"])
                continue

            # 円弧に隣接する短い区間が同じ円に乗るなら円弧に吸収させる
            if prev["type"] == "円弧" or p["type"] == "円弧":
                c = fit_circle(sub)
                if c is not None and arc_error(sub, c) <= tol * 1.5:
                    merged[-1] = make_arc(points, prev["i0"], p["i1"], c)
                    continue

        merged.append(p)
    return merged


def segment_stroke(points, epsilon, tol):
    """
    ストロークを「線分」と「円弧」の列に分解する。
    RDPで求めた頂点を切れ目の候補として、各位置から一番長く伸ばせる形状を貪欲に選ぶ。
    """
    idx = douglas_peucker_indices(points, epsilon)
    prims = []

    i = 0
    while i < len(idx) - 1:
        # --- 直線として何区間まで伸ばせるか ---
        j_line = i + 1
        for j in range(i + 1, len(idx)):
            if line_error(points[idx[i] : idx[j] + 1]) <= tol:
                j_line = j
            else:
                break

        # --- 円弧として何区間まで伸ばせるか（円弧は最低2区間必要） ---
        j_arc, arc_fit = i, None
        for j in range(i + 2, len(idx)):
            sub = points[idx[i] : idx[j] + 1]
            c = fit_circle(sub)
            if c is None:
                break
            span = float(np.hypot(*(sub[-1] - sub[0])))
            # 半径が大きすぎる＝ほぼ直線なので円弧とはみなさない
            if c[2] > max(span, 1.0) * 20:
                break
            if arc_error(sub, c) <= tol:
                j_arc, arc_fit = j, c
            else:
                break

        # 長く説明できたほうを採用（同じなら線分を優先）
        if j_arc > j_line and arc_fit is not None:
            prims.append(make_arc(points, idx[i], idx[j_arc], arc_fit))
            i = j_arc
        else:
            prims.append(make_line(points, idx[i], idx[j_line]))
            i = j_line

    return idx, merge_primitives(points, prims, tol)


def primitive_xy(p, n=96):
    """図形を描くための座標列を返す"""
    if p["type"] == "線分":
        return np.array([p["start"], p["end"]], dtype=float)
    t = np.linspace(p["a0"], p["a1"], n)
    cx, cy = p["center"]
    return np.column_stack([cx + p["radius"] * np.cos(t), cy + p["radius"] * np.sin(t)])


# ==========================================================
# 清書結果をキャンバスに戻すための fabric.js オブジェクト生成
# ==========================================================
def to_fabric_path(xy, color, stroke_width):
    """点列を fabric.js の path オブジェクト（辞書）に変換する"""
    xy = np.asarray(xy, dtype=float)
    cmds = [["M", float(xy[0, 0]), float(xy[0, 1])]]
    cmds += [["L", float(x), float(y)] for x, y in xy[1:]]

    minx, miny = xy.min(axis=0)
    maxx, maxy = xy.max(axis=0)
    return {
        "type": "path",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": float(minx),
        "top": float(miny),
        "width": float(maxx - minx),
        "height": float(maxy - miny),
        "fill": None,
        "stroke": color,
        "strokeWidth": stroke_width,
        "strokeDashArray": None,
        "strokeLineCap": "round",
        "strokeDashOffset": 0,
        "strokeLineJoin": "round",
        "strokeUniform": False,
        "strokeMiterLimit": 10,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "shadow": None,
        "visible": True,
        "backgroundColor": "",
        "fillRule": "nonzero",
        "paintFirst": "fill",
        "globalCompositeOperation": "source-over",
        "skewX": 0,
        "skewY": 0,
        "path": cmds,
    }


def beautify(json_data, epsilon, tol, stroke_width, colorize):
    """キャンバスの全ストロークを清書し、(新しい初期描画JSON, 全図形リスト) を返す"""
    objects, all_prims = [], []
    for obj in (json_data or {}).get("objects", []):
        if str(obj.get("type", "")).lower() != "path":
            objects.append(obj)  # path 以外（画像など）はそのまま残す
            continue

        pts = path_to_points(obj.get("path", []))
        if len(pts) < 2:
            continue

        _, prims = segment_stroke(pts, epsilon, tol)
        all_prims.extend(prims)
        for p in prims:
            color = (LINE_COLOR if p["type"] == "線分" else ARC_COLOR) if colorize else "#000000"
            objects.append(to_fabric_path(primitive_xy(p), color, stroke_width))

    return {"version": "4.4.0", "objects": objects}, all_prims


# ==========================================================
# UI
# ==========================================================
st.sidebar.header("パラメータ")
epsilon = st.sidebar.slider("epsilon（RDPの閾値）", 0.0, 50.0, 5.0, 0.1)
tol = st.sidebar.slider("あてはめ許容誤差 (px)", 0.5, 30.0, 5.0, 0.5)
stroke_width = st.sidebar.slider("線の太さ", 1, 10, 3)
colorize = st.sidebar.checkbox("種類ごとに色分け（青=線分 / 緑=円弧）", True)
auto = st.sidebar.checkbox("描き終わったら自動で清書", True)
mode = st.sidebar.selectbox("描画モード", ["freedraw", "transform"], help="transform では図形の移動・削除ができます")

# セッション状態の初期化
st.session_state.setdefault("drawing", None)   # キャンバスに読み込ませる図形
st.session_state.setdefault("canvas_id", 0)    # 図形を差し替えるためのキー
st.session_state.setdefault("n_shapes", 0)     # 清書済み図形の数
st.session_state.setdefault("prims", [])       # 判定結果

c1, c2 = st.columns([1, 2])
if c1.button("消去"):
    st.session_state.drawing = {"version": "4.4.0", "objects": []}
    st.session_state.canvas_id += 1
    st.session_state.n_shapes = 0
    st.session_state.prims = []
    st.rerun()

# キャンバスの設定
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",   # 塗りつぶしなし
    stroke_width=stroke_width,       # 線の太さ
    stroke_color="#000000",          # 手書き線の色
    background_color="#FFFFFF",      # 背景色
    background_image=None,           # 背景画像（なし）
    update_streamlit=True,           # Streamlitをリアルタイムで更新
    width=CANVAS_W,
    height=CANVAS_H,
    drawing_mode=mode,               # 描画モード
    initial_drawing=st.session_state.drawing,
    key=f"canvas_{st.session_state.canvas_id}",
)

objects = (canvas_result.json_data or {}).get("objects", [])

# 新しいストロークが増えたら（または「清書する」が押されたら）清書を実行
should_clean = st.session_state.pop("force_clean", False) or (
    auto and len(objects) > st.session_state.n_shapes
)

if should_clean and objects:
    drawing, prims = beautify(canvas_result.json_data, epsilon, tol, stroke_width, colorize)
    st.session_state.drawing = drawing
    st.session_state.prims = prims
    st.session_state.n_shapes = len(drawing["objects"])
    st.session_state.canvas_id += 1  # キーを変えてキャンバスを描き直す
    st.rerun()

# ==========================================================
# 判定結果の表示
# ==========================================================
prims = st.session_state.prims
if prims:
    st.subheader(f"判定結果：{len(prims)} 個の図形")
    rows = []
    for k, p in enumerate(prims, 1):
        if p["type"] == "線分":
            detail = f"長さ {p['length']:.1f} px"
        else:
            detail = f"半径 {p['radius']:.1f} px / 中心角 {abs(np.degrees(p['a1'] - p['a0'])):.0f}°"
        rows.append({
            "#": k,
            "種類": p["type"],
            "始点": f"({p['start'][0]:.0f}, {p['start'][1]:.0f})",
            "終点": f"({p['end'][0]:.0f}, {p['end'][1]:.0f})",
            "詳細": detail,
            "最大誤差(px)": round(p["error"], 2),
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)
else:
    st.info("キャンバスに線を描いてください。描き終わると自動で清書されます。")

# 元のストロークと簡略化点の比較（参考表示）
with st.expander("元のストロークとRDPの結果を見る"):
    raw = [path_to_points(o.get("path", [])) for o in objects
           if str(o.get("type", "")).lower() == "path"]
    raw = [p for p in raw if len(p) >= 2]
    if raw:
        fig, ax = plt.subplots(figsize=(6, 4))
        for pts in raw:
            point_list = [tuple(p) for p in pts]
            simplified = np.array(
                douglas_peucker(point_list, 0, len(point_list), epsilon), dtype=float
            )
            ax.plot(pts[:, 0], pts[:, 1], "-", color="0.8", lw=1)
            ax.plot(simplified[:, 0], simplified[:, 1], "o-", ms=4, color="red", lw=1)
            st.write(f"元の点の数: {len(pts)} → 簡略化後: {len(simplified)}")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # キャンバスは下向きが +y なので上下を合わせる
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(CANVAS_H, 0)
        st.pyplot(fig)
        plt.close(fig)  # 図を閉じてメモリリークを防ぐ
    else:
        st.write("まだ描かれていません。")

with st.expander("描画結果のJSONデータ"):
    st.json(canvas_result.json_data)


