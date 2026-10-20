"""Interface de démonstration « Accueil au guichet » : réponse, sources, trace (outils, tokens, latence).

    uv run --extra ui streamlit run app/streamlit_app.py
"""
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import Assistant  # noqa: E402

st.set_page_config(page_title="Assistant démarches", page_icon="🛎️", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- couleurs
ACCENT = "#2340D0"   # bleu du guichet
COUNTER = "#D6455D"  # rouge du comptoir
INK = "#141B4D"
MUTED = "#5B6286"
LINE = "#CFD6F0"
SOFT = "#EEF1FB"
PAPER = "#FFFDF7"
BG = "#EAEFFD"

# --------------------------------------------------------------------------- style global
st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Caveat:wght@500;600&display=swap');

html, body, .stApp, .stMarkdown, button, input, textarea, label, p {{
  font-family: Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}}
.stApp {{ background: {BG}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 1.2rem !important; max-width: 1240px; }}

/* ----- animations de la scène ----- */
.blink{{transform-box:fill-box;transform-origin:center;animation:blink 5.2s infinite}}
@keyframes blink{{0%,93%,100%{{transform:scaleY(1)}}95.5%{{transform:scaleY(.08)}}}}
.breath{{transform-box:fill-box;transform-origin:50% 100%;animation:breath 5s ease-in-out infinite}}
@keyframes breath{{0%,100%{{transform:scaleY(1)}}50%{{transform:scaleY(1.01)}}}}
.wave{{transform-origin:620px 450px;animation:wave 3.4s ease-in-out .7s 1 both}}
@keyframes wave{{0%{{transform:rotate(-34deg)}}12%{{transform:rotate(6deg)}}28%{{transform:rotate(-16deg)}}44%{{transform:rotate(6deg)}}60%{{transform:rotate(-14deg)}}85%,100%{{transform:rotate(0deg)}}}}
.leaf{{transform-box:fill-box;transform-origin:50% 100%;animation:sway 6s ease-in-out infinite}}
.leaf.b{{animation-delay:-2s}}.leaf.c{{animation-delay:-4s}}
@keyframes sway{{0%,100%{{transform:rotate(-2.5deg)}}50%{{transform:rotate(3deg)}}}}
.tick{{transform-origin:1060px 150px;animation:tick 60s steps(60) infinite}}
@keyframes tick{{to{{transform:rotate(360deg)}}}}
.pop{{transform-box:fill-box;transform-origin:0% 100%;animation:pop .55s cubic-bezier(.2,1.4,.4,1) .6s both}}
@keyframes pop{{from{{opacity:0;transform:scale(.7)}}to{{opacity:1;transform:scale(1)}}}}
.w{{opacity:0;animation:wordin .35s ease-out forwards}}
@keyframes wordin{{from{{opacity:0}}to{{opacity:1}}}}
.lampsway{{animation:lamp 7s ease-in-out infinite}}
@keyframes lamp{{0%,100%{{transform:rotate(-.8deg)}}50%{{transform:rotate(.8deg)}}}}
.rise{{animation:rise .55s cubic-bezier(.2,.9,.3,1) both}}
@keyframes rise{{from{{opacity:0;transform:translateY(60px) rotate(-2deg)}}to{{opacity:1;transform:translateY(0) rotate(-.4deg)}}}}
.stamp{{animation:stamp .4s cubic-bezier(.2,1.6,.4,1) .35s both}}
@keyframes stamp{{from{{opacity:0;transform:rotate(-12deg) scale(2)}}to{{opacity:1;transform:rotate(-12deg) scale(1)}}}}
@media (prefers-reduced-motion:reduce){{*{{animation-duration:.01ms!important;animation-delay:0s!important;animation-iteration-count:1!important}}.w{{opacity:1}}}}

/* ----- bandeau d'infos ----- */
.infobar {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px 20px; font-size:13.5px;
           color:{MUTED}; padding:0 6px 10px; }}
.infobar b {{ color:{INK}; font-weight:700; }}
.infobar .tag {{ color:{COUNTER}; font-weight:600; }}

/* ----- scène ----- */
.scene {{ position:relative; border-radius:24px; overflow:hidden; height:clamp(380px, 50vw, 600px);
          box-shadow:0 18px 40px rgba(27,42,138,.14); background:#eef2ff; }}
.scene svg {{ position:absolute; inset:0; width:100%; height:100%; display:block; }}

/* bouton « guichet » posé sur la scène */
.st-key-open_guichet {{ position:relative; z-index:3; margin-top:-104px; margin-bottom:56px;
                        width:100% !important; display:flex; justify-content:center; }}
.st-key-open_guichet .stButton {{ width:auto; }}
.st-key-open_guichet button {{
  background:#fff !important; color:{ACCENT} !important; border:3px solid {ACCENT} !important;
  border-radius:999px !important; padding:14px 34px !important; min-height:58px;
  box-shadow:0 10px 24px rgba(27,42,138,.18); animation:pulse 2.4s ease-in-out 2.2s infinite;
}}
.st-key-open_guichet button p {{ font-size:19px !important; font-weight:700 !important; }}
.st-key-open_guichet button:hover {{ background:{ACCENT} !important; color:#fff !important; }}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.04)}}}}

/* ----- onglets ----- */
[data-testid="stTabs"] [role="tablist"] {{ gap:26px; border-bottom:1.5px solid {LINE}; }}
[data-testid="stTab"] {{ padding:10px 2px; }}
[data-testid="stTab"] p {{ font-size:18px !important; font-weight:700; color:{MUTED}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color:{ACCENT} !important; }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background:{ACCENT}; height:3px; border-radius:3px; }}

/* ----- boutons ----- */
button[data-testid="stBaseButton-primary"] {{
  background:{ACCENT}; border:0; border-radius:999px; padding:10px 28px; min-height:48px;
  box-shadow:0 6px 16px rgba(35,64,208,.35);
}}
button[data-testid="stBaseButton-primary"] p {{ font-weight:700; font-size:16px; }}
button[data-testid="stBaseButton-secondary"] {{ border-radius:12px; border:1.5px solid {LINE}; background:#fff; }}
button[data-testid="stBaseButton-tertiary"] p {{ color:{MUTED}; font-weight:600; }}
button[data-variant="segmented_control"] {{ min-height:44px; padding:8px 18px; background:#fff;
  border:1.5px solid {LINE} !important; color:{INK}; }}
button[data-variant="segmented_control"][aria-checked="true"] {{ background:{ACCENT} !important;
  border-color:{ACCENT} !important; color:#fff !important; }}
button[data-variant="segmented_control"] p {{ font-weight:600; font-size:15px; }}
button[data-variant="pills"] {{ background:{SOFT}; border:0 !important; border-radius:999px; }}

/* ----- feuille de demande (dialogue) ----- */
div[role="dialog"] {{
  background:{PAPER} !important; border-radius:6px 6px 10px 10px !important;
  box-shadow:0 2px 3px rgba(20,27,77,.12), 0 30px 70px rgba(20,27,77,.38) !important;
  overflow:visible !important;
}}
div[role="dialog"]::before {{ content:""; position:absolute; left:50%; top:-14px; margin-left:-44px; width:88px;
  height:28px; border-radius:8px; background:{ACCENT}; box-shadow:0 3px 6px rgba(20,27,77,.3); }}
div[role="dialog"]::after {{ content:""; position:absolute; right:0; top:0; width:34px; height:34px;
  background:linear-gradient(225deg,#e8e3d3 50%,rgba(0,0,0,0) 50%); border-radius:0 6px 0 0; }}
div[role="dialog"] h2 {{ font-size:32px !important; font-weight:700 !important; letter-spacing:-.8px; color:{INK}; }}
div[role="dialog"] label p {{ font-size:15px !important; font-weight:700 !important; color:{INK}; }}
.sheet-kicker {{ font-size:14px; font-weight:600; color:{MUTED}; margin:0; }}
.sheet-lead {{ font-size:16px; color:{MUTED}; margin:0 0 6px; line-height:1.5; }}
.hint {{ font-size:14px; color:{MUTED}; margin:-6px 0 6px; }}
.st-key-d_q textarea {{
  background-color:transparent !important; border-radius:0 !important;
  background-image:repeating-linear-gradient(transparent,transparent 35px,#C9D2F2 35px,#C9D2F2 36px);
  background-attachment:local; font-family:Caveat,'Segoe Print',cursive !important; font-size:27px !important;
  line-height:36px !important; font-weight:600; color:#1D2BB8 !important; padding:0 4px !important;
}}
.st-key-d_q [data-testid="stTextAreaRootElement"] {{ background:transparent; border:0; border-top:1px solid #C9D2F2;
  border-bottom:2px solid #C9D2F2; border-radius:0; }}
.st-key-d_adv {{ background:#F4F6FD; border-radius:14px; }}

/* ----- réponse ----- */
.reply {{ position:relative; background:{PAPER}; border-radius:6px 6px 10px 10px; padding:34px 38px 26px;
          margin:30px 0 8px; box-shadow:0 2px 3px rgba(20,27,77,.12), 0 18px 40px rgba(20,27,77,.18); }}
.reply::before {{ content:""; position:absolute; left:50%; top:-12px; margin-left:-40px; width:80px; height:24px;
                  border-radius:7px; background:{ACCENT}; box-shadow:0 3px 6px rgba(20,27,77,.3); }}
.reply .kick {{ font-size:13.5px; font-weight:600; color:{MUTED}; margin:0 0 2px; }}
.reply .q {{ font-family:Caveat,'Segoe Print',cursive; font-size:30px; font-weight:600; color:#1D2BB8;
             line-height:1.15; margin:0 230px 16px 0; }}
.reply .a {{ font-size:16.5px; line-height:1.65; color:{INK}; border-top:1.5px dashed {LINE}; padding-top:14px; }}
.reply .a ul {{ margin:6px 0 6px 20px; padding:0; }}
.stampbox {{ position:absolute; right:28px; top:26px; pointer-events:none; }}
.stampbox div {{ border:4px solid; border-radius:14px; padding:6px 16px; font-size:19px; font-weight:800;
                 letter-spacing:.4px; background:rgba(255,253,247,.85); transform:rotate(-12deg); }}
.src-title {{ font-size:14px; font-weight:700; color:{INK}; margin:20px 0 8px; }}
.src-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:10px; }}
.src {{ display:block; text-decoration:none !important; background:#fff; border:1.5px solid {LINE};
        border-radius:12px; padding:12px 14px; transition:border-color .15s, transform .15s; }}
.src:hover {{ border-color:{ACCENT}; transform:translateY(-2px); }}
.src .t {{ font-size:15px; font-weight:600; color:{ACCENT}; line-height:1.35; }}
.src .m {{ font-size:12.5px; color:{MUTED}; margin-top:5px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px 8px; margin-top:18px; }}
.chips span {{ background:{SOFT}; border-radius:999px; padding:5px 12px; font-size:12.5px; color:{MUTED}; }}
.chips b {{ color:{INK}; font-weight:600; }}
.empty {{ text-align:center; color:{MUTED}; font-size:16px; padding:22px 0 8px; }}

/* ----- évaluation ----- */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; margin:8px 0 18px; }}
.kpi {{ background:#fff; border-radius:16px; padding:16px 18px; box-shadow:0 4px 12px rgba(27,42,138,.08); }}
.kpi .v {{ font-size:30px; font-weight:800; color:{ACCENT}; letter-spacing:-.5px; }}
.kpi .l {{ font-size:13px; color:{MUTED}; margin-top:2px; }}
.sec-title {{ font-size:20px; font-weight:700; color:{INK}; margin:14px 0 6px; }}
.foot {{ font-size:12px; color:{MUTED}; margin-top:40px; padding-top:12px; border-top:1px solid {LINE}; }}

@media (max-width:700px) {{
  .reply {{ padding:30px 20px 20px; }}
  .reply .q {{ margin-right:0; margin-top:44px; }}
  .stampbox {{ right:16px; top:20px; }}
}}
</style>
""")


# --------------------------------------------------------------------------- scène
def scene_svg(line1: str, line2: str, line3: str) -> str:
    a, c = ACCENT, COUNTER
    words = []
    delay = 1.1
    for y, size, weight, fill, text in ((300, 46, 700, INK, line1), (344, 26, 500, "#4A5280", line2),
                                        (378, 26, 500, "#4A5280", line3)):
        spans = []
        for wd in text.split():
            spans.append(f'<tspan class="w" style="animation-delay:{delay:.2f}s">{html.escape(wd)}</tspan>')
            delay += 0.15
        words.append(f'<text x="904" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                     f'letter-spacing="{-1 if size > 30 else 0}">{" ".join(spans)}</text>')
    bubble = "".join(words)
    return f"""
<div class="scene">
<svg viewBox="0 0 1440 900" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
<defs>
<linearGradient id="wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f6f8ff"/><stop offset="1" stop-color="#e6ecff"/></linearGradient>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#b9d1ff"/><stop offset="1" stop-color="#eaf2ff"/></linearGradient>
<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#dbe1f6"/><stop offset="1" stop-color="#cdd5f0"/></linearGradient>
<filter id="soft" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#1b2a8a" flood-opacity=".16"/></filter>
<filter id="tiny" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#1b2a8a" flood-opacity=".18"/></filter>
</defs>
<rect x="0" y="0" width="1440" height="720" fill="url(#wall)"/>
<rect x="0" y="700" width="1440" height="22" fill="#d3dbf5"/>
<rect x="0" y="722" width="1440" height="178" fill="url(#floor)"/>
<g stroke="#c3ccec" stroke-width="2" opacity=".7"><line x1="0" y1="780" x2="1440" y2="780"/><line x1="0" y1="846" x2="1440" y2="846"/></g>
<g>
<path d="M130 430 V230 a110 110 0 0 1 220 0 V430 Z" fill="#fff" filter="url(#tiny)"/>
<path d="M144 418 V230 a96 96 0 0 1 192 0 V418 Z" fill="url(#sky)"/>
<ellipse cx="206" cy="270" rx="46" ry="15" fill="#fff" opacity=".95"/><ellipse cx="236" cy="258" rx="30" ry="14" fill="#fff" opacity=".95"/>
<ellipse cx="290" cy="330" rx="38" ry="12" fill="#fff" opacity=".85"/>
<line x1="240" y1="134" x2="240" y2="418" stroke="#fff" stroke-width="8"/><line x1="144" y1="300" x2="336" y2="300" stroke="#fff" stroke-width="8"/>
<rect x="116" y="428" width="248" height="16" rx="8" fill="#fff" filter="url(#tiny)"/>
</g>
<g class="lampsway" style="transform-origin:520px 0">
<line x1="520" y1="0" x2="520" y2="108" stroke="#141b4d" stroke-width="3"/>
<path d="M470 158 C470 120 570 120 570 158 Z" fill="#2340d0"/><ellipse cx="520" cy="160" rx="50" ry="8" fill="#fff6d8"/>
</g>
<g class="lampsway" style="transform-origin:920px 0;animation-delay:-3s">
<line x1="920" y1="0" x2="920" y2="108" stroke="#141b4d" stroke-width="3"/>
<path d="M870 158 C870 120 970 120 970 158 Z" fill="#2340d0"/><ellipse cx="920" cy="160" rx="50" ry="8" fill="#fff6d8"/>
</g>
<circle cx="1060" cy="150" r="46" fill="#fff" stroke="#2340d0" stroke-width="6" filter="url(#tiny)"/>
<g stroke="#141b4d" stroke-width="4" stroke-linecap="round"><line x1="1060" y1="150" x2="1060" y2="122"/><line x1="1060" y1="150" x2="1080" y2="162"/></g>
<line class="tick" x1="1060" y1="150" x2="1060" y2="116" stroke="#d6455d" stroke-width="2.5" stroke-linecap="round"/>
<circle cx="1060" cy="150" r="5" fill="#2340d0"/>
<g filter="url(#tiny)"><rect x="1196" y="96" width="156" height="150" rx="14" fill="#fff"/></g>
<rect x="1214" y="116" width="64" height="12" rx="6" fill="{a}"/>
<rect x="1214" y="144" width="118" height="8" rx="4" fill="#cdd6f6"/><rect x="1214" y="164" width="96" height="8" rx="4" fill="#cdd6f6"/><rect x="1214" y="184" width="108" height="8" rx="4" fill="#cdd6f6"/><rect x="1214" y="204" width="70" height="8" rx="4" fill="#cdd6f6"/>
<circle cx="1336" cy="102" r="9" fill="{c}"/>
<ellipse cx="720" cy="852" rx="520" ry="26" fill="#1b2a8a" opacity=".13"/>
<g class="breath">
<path d="M822 452 C836 484 842 512 838 548" stroke="{a}" stroke-width="46" stroke-linecap="round" fill="none"/>
<circle cx="838" cy="552" r="25" fill="#f2c9a5"/>
<path d="M606 570 V478 C606 430 642 408 682 402 H758 C798 408 834 430 834 478 V570 Z" fill="{a}"/>
<path d="M682 402 L720 478 L758 402 Z" fill="#fff"/>
<path d="M682 402 L708 480 L644 446 Z" fill="#1a2fa6"/><path d="M758 402 L732 480 L796 446 Z" fill="#1a2fa6"/>
<path d="M720 480 L707 512 L720 540 L733 512 Z" fill="{c}"/>
<rect x="772" y="498" width="62" height="30" rx="6" fill="#fff"/><rect x="772" y="498" width="62" height="8" rx="4" fill="#8fb2ee"/>
<text x="803" y="522" text-anchor="middle" font-size="13" font-weight="700" fill="#141b4d">Lexi</text>
<rect x="698" y="366" width="44" height="46" rx="18" fill="#e9b58f"/>
<circle cx="720" cy="322" r="64" fill="#f2c9a5"/>
<path d="M650 334 C638 248 690 218 732 222 C786 226 810 284 792 338 C790 306 776 284 750 268 C726 292 692 302 660 302 C654 312 651 322 650 334 Z" fill="#2b2d42"/>
<path d="M742 250 C722 274 694 288 668 292" stroke="#4a4d72" stroke-width="4" fill="none" stroke-linecap="round" opacity=".6"/>
<path d="M760 244 C780 254 788 272 788 292" stroke="#4a4d72" stroke-width="4" fill="none" stroke-linecap="round" opacity=".5"/>
<path d="M654 324 C646 350 650 374 662 386 L680 374 C670 358 666 342 668 324 Z" fill="#2b2d42"/>
<path d="M786 324 C794 350 790 374 778 386 L760 374 C770 358 774 342 772 324 Z" fill="#2b2d42"/>
<ellipse cx="682" cy="350" rx="11" ry="7" fill="#f19a8e" opacity=".5"/><ellipse cx="758" cy="350" rx="11" ry="7" fill="#f19a8e" opacity=".5"/>
<g class="blink"><circle cx="696" cy="330" r="6" fill="#1d1d1f"/><circle cx="744" cy="330" r="6" fill="#1d1d1f"/></g>
<path d="M684 312 Q696 304 708 311" stroke="#2b2d42" stroke-width="3.5" fill="none" stroke-linecap="round"/><path d="M732 311 Q744 304 756 312" stroke="#2b2d42" stroke-width="3.5" fill="none" stroke-linecap="round"/>
<path d="M700 358 Q720 378 740 358" stroke="#1d1d1f" stroke-width="4" stroke-linecap="round" fill="none"/>
<path d="M652 322 C646 232 794 232 788 322" stroke="#141b4d" stroke-width="8" fill="none" stroke-linecap="round"/>
<rect x="638" y="306" width="22" height="52" rx="11" fill="#141b4d"/><rect x="780" y="306" width="22" height="52" rx="11" fill="#141b4d"/>
<path d="M790 352 Q800 396 752 392" stroke="#141b4d" stroke-width="5" fill="none" stroke-linecap="round"/>
<circle cx="746" cy="392" r="9" fill="#141b4d"/>
<g class="wave">
<path d="M618 452 L 540 346" stroke="{a}" stroke-width="46" stroke-linecap="round" fill="none"/>
<circle cx="534" cy="330" r="26" fill="#f2c9a5"/>
</g>
</g>
<g filter="url(#soft)"><rect x="310" y="590" width="820" height="260" rx="20" fill="#fff" stroke="{a}" stroke-width="5"/></g>
<rect x="270" y="562" width="900" height="30" rx="15" fill="{c}"/>
<g stroke="#dde3f7" stroke-width="3"><line x1="470" y1="610" x2="470" y2="830"/><line x1="970" y1="610" x2="970" y2="830"/></g>
<rect x="506" y="640" width="428" height="96" rx="22" fill="{a}"/>
<text x="720" y="699" text-anchor="middle" fill="#fff" font-size="34" font-weight="700" letter-spacing="-.5">Assistant démarches</text>
<g>
<g class="leaf a"><path d="M360 536 C336 520 338 488 364 480 C376 506 374 524 360 536 Z" fill="#8fb2ee" stroke="#141b4d" stroke-width="3"/></g>
<g class="leaf b"><path d="M366 536 C372 510 394 494 416 500 C412 522 394 534 366 536 Z" fill="#b3cdf6" stroke="#141b4d" stroke-width="3"/></g>
<g class="leaf c"><path d="M364 536 C358 508 368 474 384 462 C396 488 386 518 364 536 Z" fill="#8fb2ee" stroke="#141b4d" stroke-width="3"/></g>
<path d="M342 532 H394 L388 562 H348 Z" fill="#fff" stroke="#141b4d" stroke-width="3.5" stroke-linejoin="round"/>
</g>
<g>
<rect x="470" y="546" width="116" height="16" rx="3" fill="#e4e9fb" stroke="#c3ccec" stroke-width="2"/>
<rect x="476" y="534" width="112" height="14" rx="3" fill="#f2f4fd" stroke="#c3ccec" stroke-width="2" transform="rotate(-2 476 534)"/>
<rect x="480" y="520" width="108" height="16" rx="3" fill="#fff" stroke="#c3ccec" stroke-width="2" transform="rotate(2 480 520)"/>
<rect x="494" y="526" width="56" height="5" rx="2.5" fill="#8fb2ee" transform="rotate(2 480 520)"/>
</g>
<path d="M966 554 C966 524 1016 524 1016 554 Z" fill="{c}"/><rect x="956" y="552" width="70" height="9" rx="4.5" fill="#141b4d"/><circle cx="991" cy="522" r="5" fill="#141b4d"/>
<g>
<rect x="1078" y="518" width="40" height="44" rx="8" fill="#fff" stroke="#141b4d" stroke-width="3.5"/>
<line x1="1090" y1="518" x2="1084" y2="486" stroke="{c}" stroke-width="5" stroke-linecap="round"/><line x1="1100" y1="518" x2="1102" y2="480" stroke="{a}" stroke-width="5" stroke-linecap="round"/><line x1="1110" y1="518" x2="1120" y2="490" stroke="#141b4d" stroke-width="5" stroke-linecap="round"/>
</g>
<g class="pop" filter="url(#tiny)">
<path d="M868 240 H1146 a30 30 0 0 1 30 30 V380 a30 30 0 0 1 -30 30 H898 a30 30 0 0 1 -30 -30 V352 L818 338 L868 318 Z" fill="#fff"/>
</g>
{bubble}
</svg>
</div>
"""


# --------------------------------------------------------------------------- données
@st.cache_data
def corpus_stats() -> dict:
    p = ROOT / "docs" / "corpus_stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def eval_metrics() -> dict:
    p = ROOT / "eval" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_resource(show_spinner="Lexi range ses fiches (chargement de l'index et des modèles)…")
def load_assistant(backend: str, ollama_model: str, retrieval_mode: str, top_k: int,
                   use_reranker: bool) -> Assistant:
    return Assistant.from_disk(get_settings(llm_backend=backend, ollama_model=ollama_model,
                                            retrieval_mode=retrieval_mode, top_k=top_k,
                                            use_reranker=use_reranker))


def doc_lookup(assistant: Assistant) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for c in getattr(assistant.retriever, "chunks", []):
        docs.setdefault(c.doc_id, {"title": c.title, "theme": c.theme, "date": c.date, "url": c.url})
    return docs


def fmt_date(d: str) -> str:
    try:
        return date.fromisoformat(d).strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return d or "—"


def md_to_html(text: str) -> str:
    """Rendu minimal (gras, listes, sauts de ligne) d'un texte échappé."""
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))
    out, in_list = [], False
    for line in t.split("\n"):
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{m.group(1)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(line + "<br>")
    if in_list:
        out.append("</ul>")
    return re.sub(r"(<br>)+$", "", "".join(out))


def status_of(resp) -> str:
    if resp.stop_reason in ("injection_blocked", "output_blocked_leak") or "injection_detected" in resp.flags:
        return "block"
    return "ok" if resp.answerable else "warn"


STAMPS = {
    "ok": ("Réponse sourcée", "#18753C"),
    "warn": ("Hors des fiches", "#B34000"),
    "block": ("Demande refusée", COUNTER),
}
BUBBLES = {
    None: ("Bonjour !", "Que puis-je faire", "pour vous ?"),
    "ok": ("Voilà !", "Votre réponse est", "juste en dessous."),
    "warn": ("Hmm…", "Je ne trouve rien", "dans mes fiches."),
    "block": ("Désolée…", "Je ne peux pas", "faire cela."),
}

BACKENDS = {"ollama": "Ollama (local)", "anthropic": "Anthropic", "extractive": "Sans modèle"}
BACKEND_HINTS = {
    "ollama": "Tourne sur votre ordinateur, sans envoyer vos questions à l'extérieur.",
    "anthropic": "Modèle en ligne : plus précis, mais il faut une clé d'accès.",
    "extractive": "Sans modèle : renvoie simplement le meilleur extrait des fiches.",
}
OLLAMA_MODELS = ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "llama3.2:3b"]
MODES = {"rag": "Réponse directe", "agent": "Agent"}
MODE_HINTS = {"rag": "Une recherche, puis la réponse.", "agent": "Plusieurs recherches, calcul de dates."}
RETRIEVALS = {"hybrid": "Hybride", "bm25": "Mots-clés", "dense": "Sens"}
EXAMPLES = {
    "Carte d'identité": "Quelle est la durée de validité d'une carte d'identité ?",
    "Contravention": "Jusqu'à quand puis-je contester une contravention reçue le 1er septembre 2026 ?",
    "Dépôt de garantie": "Mon propriétaire peut-il garder mon dépôt de garantie ?",
}

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("cfg", {"backend": "ollama", "ollama_model": OLLAMA_MODELS[0], "mode": "rag", "since": None,
                      "retrieval": "hybrid", "rerank": True, "top_k": 5})


# --------------------------------------------------------------------------- feuille de demande
def _fill_example():
    label = ss.get("d_ex")
    if label:
        ss.d_q = EXAMPLES[label]


def _submit(new_cfg: dict):
    q = (ss.get("d_q") or "").strip()
    if not q:
        ss.d_err = True
        return
    ss.cfg = new_cfg
    ss.pending = q
    ss.d_q = ""
    ss.d_ex = None


@st.dialog("Votre demande", width="large")
def demand_sheet():
    cfg = ss.cfg
    st.html("<p class='sheet-kicker'>Formulaire de demande</p>"
            "<p class='sheet-lead'>Choisissez vos réglages, puis écrivez votre question sur la feuille.</p>")

    backend = st.segmented_control("Modèle", list(BACKENDS), format_func=BACKENDS.get, default=cfg["backend"],
                                   required=True, key="d_backend")
    st.html(f"<p class='hint'>{BACKEND_HINTS[backend]}</p>")
    ollama_model = cfg["ollama_model"]
    if backend == "ollama":
        ollama_model = st.selectbox("Modèle Ollama", OLLAMA_MODELS, index=OLLAMA_MODELS.index(cfg["ollama_model"]),
                                    key="d_ollama")

    mode = st.segmented_control("Mode", list(MODES), format_func=MODES.get, default=cfg["mode"], required=True,
                                key="d_mode")
    st.html(f"<p class='hint'>{MODE_HINTS[mode]}</p>")

    since = st.date_input("Fiches modifiées depuis", value=cfg["since"], format="DD/MM/YYYY", key="d_since",
                          help="Laissez vide pour chercher dans toutes les fiches.")

    with st.expander("Réglages avancés", key="d_adv"):
        retrieval = st.segmented_control("Recherche des fiches", list(RETRIEVALS), format_func=RETRIEVALS.get,
                                         default=cfg["retrieval"], required=True, key="d_retrieval")
        rerank = st.toggle("Reranker — réordonne les passages trouvés", value=cfg["rerank"], key="d_rerank")
        top_k = st.number_input("Passages lus — nombre d'extraits donnés au modèle", 1, 10, cfg["top_k"],
                                key="d_topk")

    ss.setdefault("d_q", "")
    st.text_area("Votre question", key="d_q", height=148,
                 placeholder="Écrivez ici, par exemple : quel est le délai pour contester une amende ?")
    st.pills("Idées", list(EXAMPLES), key="d_ex", on_change=_fill_example)

    c_cancel, c_send = st.columns([3, 2], vertical_alignment="center")
    if c_cancel.button("Annuler", type="tertiary"):
        st.rerun()
    c_send.button("Déposer ma demande", type="primary", width="stretch", on_click=_submit,
                  args=({"backend": backend, "ollama_model": ollama_model, "mode": mode, "since": since,
                         "retrieval": retrieval, "rerank": rerank, "top_k": int(top_k)},))
    if ss.pop("d_err", False):
        st.warning("Écrivez d'abord votre question sur la feuille.")
    if ss.get("pending"):
        st.rerun()


# --------------------------------------------------------------------------- traitement d'une demande
error_msg = None
if ss.get("pending"):
    question = ss.pop("pending")
    cfg = ss.cfg
    try:
        assistant = load_assistant(cfg["backend"], cfg["ollama_model"], cfg["retrieval"], cfg["top_k"], cfg["rerank"])
        with st.spinner("Lexi consulte les fiches et rédige votre réponse…"):
            resp = assistant.ask(question, mode=cfg["mode"],
                                 since=cfg["since"].isoformat() if cfg["since"] else None)
        ss.history.append((resp, doc_lookup(assistant)))
    except Exception as e:  # noqa: BLE001
        hint = ""
        if cfg["backend"] == "ollama":
            hint = (" Vérifiez qu'Ollama tourne (`ollama serve`) et que le modèle est téléchargé "
                    f"(`ollama pull {cfg['ollama_model']}`), ou choisissez « Sans modèle ».")
        elif cfg["backend"] == "anthropic":
            hint = " Vérifiez `ANTHROPIC_API_KEY` dans `.env` et lancez `uv sync --extra api`."
        error_msg = f"La demande n'a pas pu être traitée : {e}.{hint}"

# --------------------------------------------------------------------------- en-tête + scène
stats = corpus_stats()
themes = [t[0].split(" - ")[0] for t in stats.get("themes", [])] or ["Papiers", "Logement", "Transports"]
n_chunks = f"{stats['n_chunks']:,}".replace(",", " ") if stats.get("n_chunks") else "—"
st.html(
    f"<div class='infobar'><span><b>{stats.get('n_documents', 500)}</b> fiches Service-Public.fr · "
    f"<b>{n_chunks}</b> passages · {', '.join(themes)} · à jour au "
    f"<b>{fmt_date(stats.get('dates', {}).get('max', ''))}</b></span>"
    "<span class='tag'>Projet étudiant · non officiel</span></div>"
)

last = status_of(ss.history[-1][0]) if ss.history else None
st.markdown(scene_svg(*BUBBLES[last]), unsafe_allow_html=True)
if st.button("Cliquez ici pour poser votre question au guichet", key="open_guichet"):
    demand_sheet()

tab_chat, tab_eval = st.tabs(["Guichet", "Évaluation"])

# --------------------------------------------------------------------------- onglet Guichet
with tab_chat:
    if error_msg:
        st.error(error_msg)
    if not ss.history:
        st.html("<div class='empty'>Aucune demande déposée pour l'instant. "
                "Cliquez sur le guichet pour poser votre première question.</div>")
    else:
        if st.button("🗑  Effacer les demandes", type="tertiary"):
            ss.history = []
            st.rerun()

    for resp, docs in reversed(ss.history):
        kind = status_of(resp)
        stamp, stamp_color = STAMPS[kind]
        cards = ""
        for doc_id in resp.cited_sources:
            d = docs.get(doc_id, {})
            url = d.get("url") or f"https://www.service-public.gouv.fr/particuliers/vosdroits/{doc_id}"
            meta = " · ".join(x for x in [doc_id, d.get("theme", "").split(" - ")[0],
                                          f"mise à jour le {fmt_date(d['date'])}" if d.get("date") else ""] if x)
            cards += (f"<a class='src' href='{html.escape(url)}' target='_blank' rel='noopener'>"
                      f"<div class='t'>{html.escape(d.get('title', doc_id))} ↗</div>"
                      f"<div class='m'>{html.escape(meta)}</div></a>")
        sources = (f"<div class='src-title'>Fiches citées ({len(resp.cited_sources)})</div>"
                   f"<div class='src-grid'>{cards}</div>") if cards else ""
        cost = f"{resp.cost_usd:.4f} $" if resp.cost_usd else "0 $ (local)"
        chips = [("Latence", f"{resp.latency_s:.2f} s"), ("Tokens", f"{resp.tokens_in} → {resp.tokens_out}"),
                 ("Coût", cost), ("Modèle", resp.model or "—"), ("Mode", MODES.get(resp.mode, resp.mode)),
                 ("Arrêt", resp.stop_reason)]
        chips_html = "".join(f"<span>{k} <b>{html.escape(str(v))}</b></span>" for k, v in chips)
        st.html(
            f"<div class='reply rise'>"
            f"<div class='stampbox'><div class='stamp' style='color:{stamp_color};border-color:{stamp_color}'>"
            f"{stamp}</div></div>"
            f"<p class='kick'>Votre demande</p>"
            f"<div class='q'>{html.escape(resp.question)}</div>"
            f"<div class='a'>{md_to_html(resp.answer)}</div>"
            f"{sources}<div class='chips'>{chips_html}</div></div>"
        )
        with st.expander("Voir la trace complète (outils, appels au modèle, passages)"):
            if resp.tool_calls:
                st.markdown("**Outils appelés**")
                for t in resp.tool_calls:
                    st.markdown(f"- 🔧 `{t.name}` {t.arguments} → {t.result_summary} · {t.latency_s:.2f} s")
            if resp.llm_calls:
                st.markdown("**Appels au modèle**")
                for c in resp.llm_calls:
                    st.markdown(f"- 🧠 {c.purpose} · {c.model} · {c.tokens_in}/{c.tokens_out} tokens · "
                                f"{c.latency_s:.2f} s · {c.stop_reason}")
            if resp.retrieved_doc_ids:
                st.markdown("**Fiches récupérées**")
                st.markdown("\n".join(f"- {did} — {docs.get(did, {}).get('title', '')}"
                                      for did in dict.fromkeys(resp.retrieved_doc_ids)))
            st.json({"timings": resp.timings, "flags": resp.flags, "retrieved": resp.retrieved_chunk_ids},
                    expanded=False)

# --------------------------------------------------------------------------- onglet Évaluation
with tab_eval:
    data = eval_metrics()
    if not data:
        st.info("Aucun résultat d'évaluation. Lancez `uv run python eval/run_eval.py --questions "
                "eval/golden.jsonl --out eval/`.")
    else:
        cfg_eval = data.get("config", {})
        splits = [s for s in ("all", "dev", "test") if s in data["metrics"]]
        split = st.segmented_control("Jeu de questions", splits, default=splits[0], required=True,
                                     format_func={"all": "Toutes", "dev": "Dev", "test": "Test"}.get)
        m = data["metrics"][split]
        r, a = m.get("retrieval", {}), m.get("answers", {})
        ab, sf, cl = m.get("abstention", {}), m.get("safety", {}), m.get("cost_latency", {})

        def pct(x):
            return f"{x * 100:.0f} %" if isinstance(x, (int, float)) else "—"

        kpis = [
            (pct(a.get("judge_accuracy")), "Réponses justes (juge LLM)"),
            (pct(r.get("hit@5")), "Bonne fiche dans le top 5"),
            (pct(ab.get("unanswerable_abstention_rate")), "Abstention hors corpus"),
            (pct(sf.get("injection_resistance_rate")), "Injections résistées"),
            (pct(a.get("citation_precision")), "Précision des citations"),
            (f"{cl.get('latency_p50_s', 0):.1f} s", "Latence médiane"),
        ]
        st.html(
            f"<div class='sec-title'>{m.get('n_questions', '—')} questions · modèle "
            f"{html.escape(str(cfg_eval.get('model', '')))} · mode {html.escape(str(cfg_eval.get('mode', '')))}</div>"
            "<div class='kpi-grid'>"
            + "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{lbl}</div></div>" for v, lbl in kpis)
            + "</div>"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.html("<div class='sec-title'>Recherche</div>")
            st.dataframe({"Métrique": ["Hit@1", "Hit@5", "Recall@5", "MRR"],
                          "Valeur": [r.get("hit@1"), r.get("hit@5"), r.get("recall@5"), r.get("mrr")]},
                         hide_index=True, width="stretch")
            st.html("<div class='sec-title'>Raisons d'arrêt</div>")
            st.bar_chart(pd.Series(m.get("stop_reasons", {}), name="questions"), color=ACCENT, horizontal=True)
        with c2:
            st.html("<div class='sec-title'>Réponses</div>")
            st.dataframe({"Métrique": ["Taux de réponse", "Faux refus", "Rappel des citations", "Rappel des nombres"],
                          "Valeur": [a.get("answer_rate"), a.get("false_refusal_rate"),
                                     a.get("citation_recall"), a.get("number_recall")]},
                         hide_index=True, width="stretch")
            st.html("<div class='sec-title'>Analyse d'erreurs</div>")
            st.bar_chart(pd.Series(m.get("errors", {}), name="questions"), color=COUNTER, horizontal=True)

        report = ROOT / "eval" / "report.md"
        if report.exists():
            with st.expander("Rapport d'évaluation complet"):
                st.markdown(report.read_text(encoding="utf-8"))

st.html("<div class='foot'>Projet étudiant (Mines Paris – PSL × Albert School), sans lien avec l'administration. "
        "Les réponses s'appuient sur les fiches pratiques Service-Public.fr publiées sur data.gouv.fr ; "
        "vérifiez toujours la fiche officielle avant une démarche.</div>")
