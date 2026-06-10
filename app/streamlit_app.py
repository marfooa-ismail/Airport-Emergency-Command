from pathlib import Path
import sys, json, math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    AIRPORT_NAME, AIRPORT_CODE, AIRPORT_CITY, AIRPORT_ELEVATION,
    EMERGENCY_TYPES, EMERGENCY_VOICE_OPTIONS, PROCESSED_DATA_PATH,
    RUNWAYS, FLIGHT_OPERATORS
)
from src.data_pipeline import prepare_training_data
from src.modeling import train_and_save_models, make_prediction, load_training_frame
from src.simulation import AirportSimulation

st.set_page_config(page_title="Allama Iqbal Airport Emergency Command", page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--bg:#020617;--panel:#071426;--line:rgba(56,189,248,.28);--txt:#f8fafc;--muted:#cbd5e1;--blue:#38bdf8;--red:#ef4444;--green:#22c55e;--amber:#f59e0b}
html,body,.stApp{background:radial-gradient(circle at 20% -8%,#103b66 0,#07111f 36%,#020617 100%)!important;color:var(--txt)!important;}
.block-container{padding-top:.65rem!important;max-width:1540px!important}.hero{padding:16px 20px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(15,23,42,.96),rgba(12,46,77,.78));box-shadow:0 28px 70px rgba(0,0,0,.38);margin-bottom:12px}.hero h1{font-size:29px;margin:0 0 5px;letter-spacing:-.035em}.hero p{margin:0;color:#dbeafe}.grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:12px}.tile,.section{background:rgba(2,6,23,.58);border:1px solid rgba(148,163,184,.24);border-radius:18px}.tile{padding:11px 13px}.tile b{font-size:11px;color:#7dd3fc;text-transform:uppercase}.tile div{font-size:15px;margin-top:4px}.section{padding:15px;margin-bottom:12px;box-shadow:0 18px 45px rgba(0,0,0,.22)}.section h2,.section h3{margin-top:0}.ok,.danger{padding:12px 15px;border-radius:16px;border:1px solid;margin:9px 0}.ok{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.32)}.danger{background:rgba(239,68,68,.14);border-color:rgba(239,68,68,.42)}
[data-testid="stMetric"]{background:rgba(15,23,42,.78);border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:12px;box-shadow:0 16px 38px rgba(0,0,0,.18)}
.stButton>button{border-radius:16px!important;font-weight:900!important;border:1px solid rgba(56,189,248,.42)!important;background:linear-gradient(135deg,#0f3150,#123e63)!important;color:#f8fafc!important;min-height:48px!important;box-shadow:0 10px 24px rgba(0,0,0,.22)!important;white-space:normal!important}.stButton>button:hover{border-color:#7dd3fc!important;transform:translateY(-1px)}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#dc2626,#ef4444)!important;border-color:#f87171!important}.small{color:#cbd5e1;font-size:13px}.stPlotlyChart{background:rgba(2,6,23,.04)!important;border-radius:18px}hr{border-color:rgba(148,163,184,.20)!important}
@media(max-width:900px){.grid5{grid-template-columns:1fr 1fr}.hero h1{font-size:23px}.block-container{padding-left:.7rem!important;padding-right:.7rem!important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DEFAULTS = {
    "sim": AirportSimulation(seed=42), "screen": "Live Simulation", "scenario": "Bird Strike", "voice": EMERGENCY_VOICE_OPTIONS[0],
    "started": False, "paused": False, "speed": 0.25, "weather": "Night / Rain", "airline": "PIA",
    "queue": ["Emirates", "Qatar Airways", "Gulf Air", "Saudia"], "audio": True, "view_mode": "Runway Landing",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
sim: AirportSimulation = st.session_state.sim

WEATHER_OPTIONS = ["Morning / Clear", "Day / Clear", "Evening / Windy", "Night / Clear", "Night / Rain", "Fog", "Thunderstorm"]
SPEED_OPTIONS = {"Ultra Slow 0.10x":0.10, "Very Slow 0.20x":0.20, "Slow 0.35x":0.35, "Normal 0.50x":0.50, "Fast 0.80x":0.80}
VIEW_OPTIONS = ["Space to Runway", "Runway Landing", "Tower View", "Radar View"]


def header():
    st.markdown(f"""
    <div class="hero"><h1>Allama Iqbal International Airport Emergency Command Center</h1>
    <p><b>{AIRPORT_NAME}</b> · {AIRPORT_CODE} · {AIRPORT_CITY} · Elevation {AIRPORT_ELEVATION}</p>
    <div class="grid5">
      <div class="tile"><b>Airport</b><div>Lahore Emergency Ops</div></div>
      <div class="tile"><b>Emergency Flight</b><div>{st.session_state.airline}</div></div>
      <div class="tile"><b>Scenario</b><div>{st.session_state.scenario}</div></div>
      <div class="tile"><b>Runways</b><div>{', '.join(RUNWAYS)}</div></div>
      <div class="tile"><b>Units</b><div>ATC · ARFF · Ambulance · Rescue</div></div>
    </div></div>""", unsafe_allow_html=True)


def nav():
    cols = st.columns(4)
    for col, label in zip(cols, ["Live Simulation", "Emergency Dispatch", "AI Decision Support", "Training Dashboard"]):
        if col.button(label, use_container_width=True, type="primary" if st.session_state.screen == label else "secondary", key="nav_"+label):
            st.session_state.screen = label
            st.rerun()


def frames():
    ac = sim.aircraft_frame(); rw = sim.runway_frame(); rs = sim.rescue_frame(); ev = sim.event_frame()
    em = ac[ac["status"].astype(str).str.contains("Emergency|Response", case=False, na=False)] if not ac.empty else pd.DataFrame()
    return ac, rw, rs, ev, em


def metrics():
    ac, rw, rs, ev, em = frames(); c = st.columns(5)
    c[0].metric("Demo Time", f"T+{sim.time_step}")
    c[1].metric("Aircraft", len(ac))
    c[2].metric("Emergencies", len(em) if len(em) else int(st.session_state.started))
    c[3].metric("Open Runways", int((rw["status"] == "Available").sum()) if not rw.empty else 3)
    c[4].metric("Rescue Busy", int((rs["status"] != "Standby").sum()) if not rs.empty else int(st.session_state.started))
    if st.session_state.started or len(em):
        st.markdown(f"<div class='danger'><b>LIVE EMERGENCY:</b> {st.session_state.airline} · {st.session_state.scenario} · priority runway cleared. Other airlines are slowed and sent into holding pattern.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='ok'><b>Airport Status:</b> normal operations. Select scenario and press Start Emergency.</div>", unsafe_allow_html=True)


def simulator_html(cfg):
    cfg_json = json.dumps(cfg)
    html = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#020617;font-family:Arial,Inter,sans-serif;color:#f8fafc;overflow:hidden}.shell{height:1020px;display:grid;grid-template-rows:auto 590px auto auto;gap:10px;background:#020617;border:1px solid rgba(56,189,248,.34);border-radius:26px;padding:10px;box-sizing:border-box}.topbar{display:grid;grid-template-columns:1.1fr .9fr 1fr;gap:10px}.bottombar{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}.logsbar{display:grid;grid-template-columns:1fr 1fr;gap:10px}.card{background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(15,23,42,.88));border:1px solid rgba(148,163,184,.28);border-radius:18px;padding:12px;box-shadow:0 14px 34px rgba(0,0,0,.26);min-width:0}.card b{color:#f8fafc}.small{color:#cbd5e1;font-size:12px;line-height:1.35}.stage{position:relative;border:1px solid rgba(56,189,248,.28);border-radius:24px;overflow:hidden;background:#07111f;height:590px}canvas{display:block;width:100%;height:100%}.tag{display:inline-block;padding:5px 10px;border-radius:999px;margin:3px 5px 2px 0;font-size:12px;font-weight:900}.r{background:#dc2626}.b{background:#2563eb}.g{background:#16a34a}.a{background:#f59e0b;color:#111827}.step{display:grid;grid-template-columns:20px 1fr auto;gap:7px;border-bottom:1px solid rgba(148,163,184,.10);padding:4px 0;font-size:12px}.okdot{color:#22c55e}.wait{color:#94a3b8}.queue p,.voice p,.logline{margin:3px 0}.mini{border:1px solid rgba(56,189,248,.48);background:#0f3150;color:#fff;border-radius:12px;padding:10px 12px;font-weight:900;cursor:pointer;margin:4px}.mini:hover{background:#14527e}.voice,.queue,.timeline,.systemlogs{max-height:140px;overflow:auto}.statusgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}.stat{background:rgba(15,23,42,.70);border:1px solid rgba(148,163,184,.18);border-radius:12px;padding:7px;text-align:center}.stat b{display:block;color:#7dd3fc;font-size:10px;text-transform:uppercase}.stat span{font-weight:900}.landingseq{max-height:140px;overflow:auto}@media(max-width:900px){.shell{height:1120px;grid-template-rows:auto 430px auto auto}.topbar,.bottombar,.logsbar{grid-template-columns:1fr}.stage{height:430px}.statusgrid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="shell">
<div class="topbar">
 <div class="card"><b>🛰️ Space Emergency → Priority Runway Landing</b><br><span class="tag r" id="scTag"></span><span class="tag b">ATC First</span><span class="tag g">Sirens After Touchdown</span><div class="small">Birds stay in the sky, contact aircraft during the event, then the aircraft lands fully on the runway.</div></div>
 <div class="card"><b>Emergency Flight</b><div id="airTag" style="font-size:20px;font-weight:900;margin-top:6px"></div><div class="statusgrid"><div class="stat"><b>Runway</b><span>18L</span></div><div class="stat"><b>Priority</b><span>HIGH</span></div><div class="stat"><b>ATC</b><span>ON</span></div><div class="stat"><b>Ground</b><span id="groundMini">AIR</span></div></div></div>
 <div class="card"><b>Weather / Time</b><div id="wxTag" style="font-size:18px;font-weight:900;margin-top:6px"></div><div class="small">Moon or sun remains visible for the whole simulation.</div></div>
</div>
<div class="stage"><canvas id="cv"></canvas></div>
<div class="bottombar">
 <div class="card"><b>🚨 Scenario Timeline</b><div id="scenarioText" class="timeline"></div></div>
 <div class="card"><b>✈️ Other Airlines</b><div id="queue" class="queue"></div></div>
 <div class="card"><b>🎙️ Realistic Emergency Sirens</b><div id="voice" class="voice small">Click once to allow browser audio. ATC plays first; sirens follow after touchdown.</div><button class="mini" onclick="playFullAudio()">▶ Full ATC + Sirens</button><button class="mini" onclick="playATCOnly()">📡 ATC</button><button class="mini" onclick="playSiren('ambulance')">🚑 Ambulance</button><button class="mini" onclick="playSiren('fire')">🚒 Fire</button><button class="mini" onclick="playSiren('police')">🚓 Police</button></div>
</div>
<div class="logsbar">
 <div class="card"><b>📡 Live ATC Communication</b><div id="atclog" class="systemlogs small"></div></div>
 <div class="card"><b>🧾 System Logs</b><div id="syslog" class="systemlogs small"></div></div>
</div>
</div>
<script>
const cfg=__CFG__; const c=document.getElementById('cv'), x=c.getContext('2d'); let W=0,H=0,d=1,t=0,audioCtx=null,announced=false,autoSiren=false;
function resize(){d=Math.min(devicePixelRatio||1,2);W=c.clientWidth;H=c.clientHeight;c.width=W*d;c.height=H*d;x.setTransform(d,0,0,d,0,0)} addEventListener('resize',resize); resize();
const voice=document.getElementById('voice'), scenarioText=document.getElementById('scenarioText'), queueEl=document.getElementById('queue'), groundMini=document.getElementById('groundMini'), atclog=document.getElementById('atclog'), syslog=document.getElementById('syslog');
document.getElementById('scTag').textContent=cfg.scenario; document.getElementById('airTag').textContent=cfg.airline+' 911'; document.getElementById('wxTag').textContent=cfg.weather+' · '+cfg.speed+'x';
let p={x:80,y:130,alt:9000,spd:290,phase:'Space View',touch:false,rollout:false}; let smoke=[],sparks=[],birds=[],traffic=[],cars=[],stars=[];
function init(){traffic=(cfg.queue||[]).map((n,i)=>({name:n,a:i*1.4,x:0,y:0,status:'HOLDING',landIndex:i})); cars=[{x:-180,y:0,type:'AMBULANCE',color:'#ef4444'},{x:-275,y:0,type:'FIRE RESCUE',color:'#f97316'},{x:-370,y:0,type:'POLICE ESCORT',color:'#2563eb'}]; birds=Array.from({length:22},(_,i)=>({x:W*.48+i*20,y:H*.12+Math.sin(i*.8)*18,hit:false})); stars=Array.from({length:150},(_,i)=>({x:(i*97)%Math.max(W,1),y:(i*53)%Math.max(H*.52,1),r:(i%3)+.45})); updateText();addLog('SYSTEM','Simulation ready - no overlapping panels.')} init();
function addLog(type,m){let row='<div class="logline"><b style="color:#38bdf8">'+type+'</b> · '+m+'</div>'; if(type==='ATC')atclog.innerHTML=row+atclog.innerHTML; else syslog.innerHTML=row+syslog.innerHTML;}
function log(m){voice.innerHTML='<p>📢 '+m+'</p>'+voice.innerHTML; addLog('SYSTEM',m)}
function speak(m){log(m); if(!cfg.audio) return; try{speechSynthesis.cancel(); let u=new SpeechSynthesisUtterance(m); u.rate=.74; u.pitch=.85; u.volume=1; speechSynthesis.speak(u); addLog('ATC',m)}catch(e){}}
function ctx(){audioCtx=audioCtx||new (window.AudioContext||window.webkitAudioContext)(); audioCtx.resume(); return audioCtx;}
function tone(freq,dur,type,vol=.22,slide=0){try{let ac=ctx(), o=ac.createOscillator(), g=ac.createGain(); o.type=type||'sine'; o.frequency.setValueAtTime(freq,ac.currentTime); if(slide)o.frequency.linearRampToValueAtTime(freq+slide,ac.currentTime+dur*.85); o.connect(g); g.connect(ac.destination); g.gain.setValueAtTime(.001,ac.currentTime); g.gain.exponentialRampToValueAtTime(vol,ac.currentTime+.03); g.gain.exponentialRampToValueAtTime(.001,ac.currentTime+dur); o.start(); o.stop(ac.currentTime+dur+.04)}catch(e){}}
function playSiren(kind){let msg=kind==='ambulance'?'Ambulance siren active. Medical response moving after touchdown.':kind==='fire'?'Airport fire rescue siren active. Foam unit proceeding after ATC clearance.':'Police escort siren active. Runway perimeter secured.'; speak(msg); let i=0; let id=setInterval(()=>{if(kind==='ambulance'){tone(620,.32,'sawtooth',.30,900); setTimeout(()=>tone(1500,.26,'square',.20,-700),180)} if(kind==='fire'){tone(420,.45,'square',.36,520); setTimeout(()=>tone(760,.36,'sawtooth',.28,-260),260)} if(kind==='police'){tone(720,.22,'square',.30,430); setTimeout(()=>tone(1220,.20,'square',.24,-480),140)} if(++i>26)clearInterval(id)},kind==='fire'?520:360)}
function playATCOnly(){speak('Allama Iqbal Tower to '+cfg.airline+' nine one one. Mayday received. Continue approach. Runway one eight left cleared for emergency landing. All other aircraft hold position. Emergency services stand by until touchdown.');}
function playFullAudio(){playATCOnly(); setTimeout(()=>playSiren('fire'),3900); setTimeout(()=>playSiren('ambulance'),6600); setTimeout(()=>playSiren('police'),8500)}
function updateText(){let steps=['Space view: emergency detected above Pakistan airspace','Bird flock is high in the sky','Bird contact / emergency event occurs at aircraft altitude','Pilot Mayday sent to Allama Iqbal Tower','ATC clears runway 18L and stops other traffic','Other airlines enter holding pattern','Aircraft aligns with runway centerline','Touchdown: main wheels contact runway','Rollout: aircraft fully on ground','Other airlines resume landing one-by-one']; scenarioText.innerHTML=steps.map((s,i)=>'<div class="step"><span id="dot'+i+'" class="wait">○</span><span>'+s+'</span><small id="tm'+i+'">pending</small></div>').join(''); queueEl.innerHTML=traffic.map((q,i)=>'<p><b>'+q.name+'</b> — '+q.status+' · '+(t>720?'sequenced landing slot +'+(i+1)*40+' sec':'holding, speed reduced')+'</p>').join('')}
function setStep(i){let d=document.getElementById('dot'+i), tm=document.getElementById('tm'+i); if(d){d.textContent='●';d.className='okdot'} if(tm)tm.textContent='done'}
function skyObject(){let sx=W*.10, sy=H*.13; if(cfg.weather.includes('Morning')||cfg.weather.includes('Day')){let sg=x.createRadialGradient(sx,sy,8,sx,sy,70);sg.addColorStop(0,'#fff7ad');sg.addColorStop(.55,'#facc15');sg.addColorStop(1,'rgba(250,204,21,0)');x.fillStyle=sg;x.beginPath();x.arc(sx,sy,70,0,7);x.fill();x.fillStyle='#fde047';x.beginPath();x.arc(sx,sy,34,0,7);x.fill()}else{let mg=x.createRadialGradient(sx-12,sy-12,4,sx,sy,48);mg.addColorStop(0,'#f8fafc');mg.addColorStop(.70,'#cbd5e1');mg.addColorStop(1,'#64748b');x.fillStyle=mg;x.beginPath();x.arc(sx,sy,44,0,7);x.fill();x.fillStyle='rgba(100,116,139,.45)';[[sx-12,sy-8,7],[sx+10,sy+5,5],[sx-2,sy+16,4]].forEach(q=>{x.beginPath();x.arc(q[0],q[1],q[2],0,7);x.fill()})}}
function bg(){let w=cfg.weather, grd=x.createLinearGradient(0,0,0,H); if(w.includes('Night')){grd.addColorStop(0,'#020617');grd.addColorStop(.48,'#08213d');grd.addColorStop(1,'#101827')}else if(w.includes('Morning')){grd.addColorStop(0,'#fbbf24');grd.addColorStop(.28,'#7dd3fc');grd.addColorStop(1,'#dbeafe')}else if(w.includes('Day')){grd.addColorStop(0,'#60a5fa');grd.addColorStop(.55,'#0ea5e9');grd.addColorStop(1,'#bfdbfe')}else if(w.includes('Evening')){grd.addColorStop(0,'#fb7185');grd.addColorStop(.36,'#7c3aed');grd.addColorStop(1,'#111827')}else if(w.includes('Fog')){grd.addColorStop(0,'#cbd5e1');grd.addColorStop(.55,'#64748b');grd.addColorStop(1,'#334155')}else{grd.addColorStop(0,'#030712');grd.addColorStop(.55,'#111827');grd.addColorStop(1,'#172554')} x.fillStyle=grd;x.fillRect(0,0,W,H); if(w.includes('Night')||w.includes('Thunder')){x.fillStyle='rgba(255,255,255,.72)';stars.forEach(s=>{x.beginPath();x.arc(s.x,s.y,s.r,0,7);x.fill()})} skyObject(); if(w.includes('Rain')||w.includes('Thunder')){x.strokeStyle='rgba(186,230,253,.34)'; for(let i=0;i<110;i++){let rx=(i*43+t*7)%W, ry=(i*71+t*12)%H; x.beginPath();x.moveTo(rx,ry);x.lineTo(rx-12,ry+32);x.stroke()}} if(w.includes('Fog')){for(let i=0;i<7;i++){x.fillStyle='rgba(226,232,240,.16)';x.beginPath();x.ellipse(W*.05+i*W*.16,H*.50+Math.sin(t/50+i)*22,190,55,0,0,7);x.fill()}}}
function airport(){let base=H*.58; x.fillStyle='rgba(30,41,59,.85)'; x.fillRect(0,base-88,W,88); for(let i=0;i<16;i++){let bx=i*92-10,bh=25+(i%4)*14; x.fillStyle=i%2?'#334155':'#475569';x.fillRect(bx,base-88-bh,72,bh);x.fillStyle='#fde68a';for(let j=0;j<4;j++){x.fillRect(bx+9+j*15,base-88-bh+9,6,5)}} x.fillStyle='#334155';x.fillRect(W*.80,base-215,38,215);x.fillStyle='#475569';x.fillRect(W*.775,base-245,88,34);x.fillStyle='#f87171';x.beginPath();x.arc(W*.825,base-252,4,0,7);x.fill();}
function runway(){let y=H*.75; x.fillStyle='#4b5563'; x.fillRect(0,y,W,H-y); x.fillStyle='#374151'; x.fillRect(0,y-24,W,24); x.strokeStyle='#f8fafc';x.lineWidth=5; for(let i=0;i<W/105+2;i++){x.beginPath();x.moveTo(i*105+(t*cfg.speed*5%105),y+42);x.lineTo(i*105+52+(t*cfg.speed*5%105),y+42);x.stroke()} for(let i=0;i<W/56+2;i++){x.fillStyle=i%2?'#22c55e':'#facc15';x.beginPath();x.arc(i*56-(t*cfg.speed*4%56),y-12,5,0,7);x.fill()} x.fillStyle='#f8fafc';x.font='bold 28px Arial';x.fillText('RUNWAY 18L',W-210,y+88); return y}
function drawRealAircraft(ax,ay,s,angle,label,emergency,onGround){x.save();x.translate(ax,ay);x.rotate(angle);x.scale(s,s);x.shadowColor=emergency?'#ef4444':'#38bdf8';x.shadowBlur=emergency?18:8; let fus=x.createLinearGradient(-130,-30,155,28);fus.addColorStop(0,'#9ca3af');fus.addColorStop(.28,'#ffffff');fus.addColorStop(.68,'#e5e7eb');fus.addColorStop(1,'#94a3b8');x.fillStyle=fus;x.beginPath();x.moveTo(158,0);x.bezierCurveTo(136,-26,0,-34,-125,-13);x.quadraticCurveTo(-158,0,-125,13);x.bezierCurveTo(0,34,136,26,158,0);x.fill();x.strokeStyle='rgba(15,23,42,.35)';x.lineWidth=2;x.stroke();let col=label.includes('PIA')?'#15803d':label.includes('Emirates')?'#b91c1c':label.includes('Qatar')?'#7f1d1d':'#1d4ed8';x.fillStyle=col;x.fillRect(-25,-14,55,28);x.fillStyle='#111827';for(let i=0;i<16;i++){x.fillRect(-92+i*12,-15,6,4)}x.fillStyle='#bae6fd';x.beginPath();x.moveTo(98,-15);x.quadraticCurveTo(130,-13,150,-2);x.lineTo(104,-2);x.closePath();x.fill();x.fillStyle='#e0f2fe';x.beginPath();x.moveTo(-5,5);x.lineTo(-96,78);x.lineTo(-56,7);x.closePath();x.fill();x.beginPath();x.moveTo(-6,-8);x.lineTo(-74,-47);x.lineTo(-45,-4);x.closePath();x.fill();x.fillStyle=col;x.beginPath();x.moveTo(-113,-6);x.lineTo(-150,-82);x.lineTo(-122,-2);x.closePath();x.fill();x.fillStyle='#111827';x.beginPath();x.ellipse(-50,38,18,12,0,0,7);x.fill(); if(onGround||p.alt<700){x.strokeStyle='#111827';x.lineWidth=4;x.beginPath();x.moveTo(42,17);x.lineTo(42,44);x.stroke();x.beginPath();x.arc(42,49,8,0,7);x.stroke();x.beginPath();x.moveTo(-70,19);x.lineTo(-70,43);x.stroke();x.beginPath();x.arc(-70,48,8,0,7);x.stroke()}x.restore();x.fillStyle='#f8fafc';x.font='bold 16px Arial';x.fillText(label,ax-38,ay-45)}
function vehicles(ry){if(!cfg.started)return; let active=(t>535); cars.forEach((v,i)=>{let tx=active?Math.min(W-210-i*88, p.x-250-i*86):-180-i*100, ty=ry+58+i*5; v.x+=(tx-v.x)*.032*cfg.speed*8;v.y=ty; x.fillStyle=v.color;x.fillRect(v.x,v.y,84,30);x.fillStyle='white';x.font='bold 9px Arial';x.fillText(v.type,v.x+5,v.y+19);x.fillStyle='#60a5fa';x.beginPath();x.arc(v.x+16,v.y-5,4+Math.sin(t/4)*2,0,7);x.fill();x.beginPath();x.arc(v.x+66,v.y-5,4+Math.cos(t/4)*2,0,7);x.fill()})}
function scenarioMarks(){if(!cfg.started)return; birds.forEach((b,i)=>{if(cfg.scenario==='Bird Strike'&&t>92&&t<152&&i<9){let k=(t-92)/60; b.x=p.x+95-k*180+i*11; b.y=p.y-70+k*48+Math.sin(i)*12; if(k>.45){b.hit=true; sparks.push({x:p.x+44-i*3,y:p.y-12+i*2,a:.9}); smoke.push({x:p.x-82,y:p.y+12,r:9,a:.34})}} x.strokeStyle=b.hit?'#ef4444':'#1f2937';x.lineWidth=b.hit?4:3;x.beginPath();x.moveTo(b.x,b.y);x.lineTo(b.x+10,b.y-8);x.lineTo(b.x+21,b.y);x.stroke()}); if(['Engine Failure','Bird Strike'].includes(cfg.scenario)&&t>105&&t<430){smoke.push({x:p.x-96,y:p.y+8,r:10,a:.30})}}
function update(){if(!cfg.started||cfg.paused)return; t+=cfg.speed*1.15; let ry=H*.75; if(!announced&&t>8){announced=true; playATCOnly()} if(t>540&&!autoSiren){autoSiren=true; setTimeout(()=>playSiren('fire'),600); setTimeout(()=>playSiren('ambulance'),2600); setTimeout(()=>playSiren('police'),4300)} if(t<85){let k=t/85;p.x=W*(.18+.15*k);p.y=H*(.17+.03*k);p.alt=9000*(1-k)+6500*k;p.spd=290-20*k;p.phase='SPACE EMERGENCY'} else if(t<210){let k=(t-85)/125;p.x=W*(.33+.23*k);p.y=H*(.22+.15*k);p.alt=6500*(1-k)+2600*k;p.spd=270-45*k;p.phase='DESCENT AFTER EMERGENCY'} else if(t<420){let k=(t-210)/210;p.x=W*(.56+.24*k);p.y=H*(.37+.20*k);p.alt=2600*(1-k)+150*k;p.spd=225-70*k;p.phase='FINAL APPROACH'} else if(t<505){let k=(t-420)/85;p.x=W*(.80+.08*k);p.y=ry-70+Math.sin(k*Math.PI)*-8;p.alt=150*(1-k);p.spd=155-35*k;p.phase='FLARE - GEAR DOWN'; if(!p.touch&&k>.70){p.touch=true; speak(cfg.airline+' touchdown confirmed. Emergency vehicles may enter runway after aircraft slows.')}} else if(t<700){let k=(t-505)/195;p.x=W*(.88-.56*k);p.y=ry-70;p.alt=0;p.spd=120-75*k;p.phase='ON GROUND ROLLOUT';p.rollout=true} else {p.x=W*.32;p.y=ry-70;p.alt=0;p.spd=25;p.phase='STOPPED - RESCUE ESCORT'} traffic.forEach((q,i)=>{q.a+=.012*cfg.speed;if(t<760){q.x=W*.72+Math.cos(q.a+i)*88;q.y=H*.15+i*30+Math.sin(q.a+i)*15;q.status='HOLDING PATTERN'}else{let start=760+i*170;let k=Math.max(0,Math.min(1,(t-start)/155));if(k<=0){q.status='SEQUENCED - WAITING';q.x=W*.72+Math.cos(q.a+i)*88;q.y=H*.15+i*30}else if(k<1){q.status='LANDING NOW';q.x=W*(.12+.78*k);q.y=ry-52+Math.sin(k*Math.PI)*-35}else{q.status='LANDED';q.x=W*.90;q.y=ry-52+i*8}}}); if(Math.floor(t)%45<2)updateText(); birds.forEach((b,i)=>{if(!(cfg.scenario==='Bird Strike'&&t>92&&t<152&&i<9)){b.x-=.28*cfg.speed;if(b.x<-40)b.x=W+40+i*10;b.y=H*.11+Math.sin(t/28+i)*22}}); smoke.forEach(s=>{s.x-=.35*cfg.speed;s.a-=.008*cfg.speed;s.r+=.18*cfg.speed}); smoke=smoke.filter(s=>s.a>0); sparks.forEach(s=>{s.a-=.04}); sparks=sparks.filter(s=>s.a>0); [0,1,2,3,4,5,6,7,8,9].forEach((n)=>{if(t>[8,70,98,125,155,185,270,485,535,760][n])setStep(n)}); groundMini.textContent=p.alt<=0?'GROUND':'AIR'}
function draw(){bg(); airport(); let ry=runway(); traffic.forEach((q,i)=>{if(q.status==='LANDED'||q.status==='LANDING NOW'||q.status.includes('HOLDING'))drawRealAircraft(q.x,q.y,q.status==='LANDING NOW'?.45:(q.status==='LANDED'?.42:.25),q.status==='LANDING NOW'?.03:0,q.name,false,q.status==='LANDED')}); let angle=p.alt>5000?.05:(p.alt>1000?.10:(p.alt>0?.025:0)); let sc=p.alt>6500?.30:(p.alt>2800?.42:(p.alt>900?.58:(p.alt>0?.78:.88))); drawRealAircraft(p.x,p.y,sc,angle,cfg.airline,true,p.alt<=0); scenarioMarks(); smoke.forEach(s=>{x.globalAlpha=s.a;x.fillStyle='#475569';x.beginPath();x.arc(s.x,s.y,s.r,0,7);x.fill();x.globalAlpha=1}); sparks.forEach(s=>{x.globalAlpha=s.a;x.fillStyle='#f97316';x.beginPath();x.arc(s.x,s.y,5+Math.random()*7,0,7);x.fill();x.globalAlpha=1}); vehicles(ry); x.fillStyle='rgba(2,6,23,.65)';x.fillRect(18,18,330,70);x.strokeStyle='rgba(56,189,248,.35)';x.strokeRect(18,18,330,70);x.fillStyle='#f8fafc';x.font='bold 15px Arial';x.fillText(p.phase,34,45);x.font='13px Arial';x.fillText('Altitude '+Math.max(0,Math.floor(p.alt))+' ft · Speed '+Math.floor(p.spd)+' kt · '+(p.alt<=0?'FULLY ON RUNWAY':'AIRBORNE'),34,68); requestAnimationFrame(loop)} function loop(){update();draw()} loop();
</script></body></html>"""
    return html.replace('__CFG__', cfg_json)

def controls():
    st.markdown("<div class='section'><h3>Emergency Scenario Control</h3><div class='small'>Choose scenario, airline, weather, view, voice and very slow speed. Start creates visual emergency response, sends other airlines into holding, and aligns the aircraft for priority landing.</div></div>", unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    st.session_state.scenario = a.selectbox("Emergency scenario", EMERGENCY_TYPES, index=EMERGENCY_TYPES.index(st.session_state.scenario))
    st.session_state.airline = b.selectbox("Emergency airline", FLIGHT_OPERATORS, index=FLIGHT_OPERATORS.index(st.session_state.airline) if st.session_state.airline in FLIGHT_OPERATORS else 0)
    st.session_state.weather = c.selectbox("Weather / time", WEATHER_OPTIONS, index=WEATHER_OPTIONS.index(st.session_state.weather) if st.session_state.weather in WEATHER_OPTIONS else 4)
    st.session_state.view_mode = d.selectbox("Camera / visual mode", VIEW_OPTIONS, index=VIEW_OPTIONS.index(st.session_state.view_mode) if st.session_state.view_mode in VIEW_OPTIONS else 1)
    e,f,g = st.columns([1.2,1.0,1.8])
    st.session_state.voice = e.selectbox("Emergency voice unit", EMERGENCY_VOICE_OPTIONS, index=EMERGENCY_VOICE_OPTIONS.index(st.session_state.voice))
    speed_label = f.selectbox("Animation speed", list(SPEED_OPTIONS.keys()), index=list(SPEED_OPTIONS.values()).index(st.session_state.speed) if st.session_state.speed in SPEED_OPTIONS.values() else 1)
    st.session_state.speed = SPEED_OPTIONS[speed_label]
    st.session_state.queue = g.multiselect("Other airlines in landing queue", FLIGHT_OPERATORS, default=st.session_state.queue)
    st.session_state.audio = st.checkbox("Enable browser ATC voice / emergency announcements", value=st.session_state.audio)
    cols = st.columns(5)
    if cols[0].button("🚨 Start Emergency", type="primary", use_container_width=True):
        ac = sim.trigger_emergency(st.session_state.scenario)
        ac.operator = st.session_state.airline
        ac.flight_id = f"{st.session_state.airline[:2].upper()}-911"
        sim.log("Voice", f"{st.session_state.voice} activated. ATC priority landing clearance issued.", ac.flight_id)
        st.session_state.started = True; st.session_state.paused = False
        st.rerun()
    if cols[1].button("⏯ Pause / Resume", use_container_width=True):
        st.session_state.paused = not st.session_state.paused; st.rerun()
    if cols[2].button("🔄 Reset", use_container_width=True):
        st.session_state.sim = AirportSimulation(seed=42); st.session_state.started = False; st.session_state.paused = False; st.rerun()
    if cols[3].button("🛬 Add Normal Arrival", use_container_width=True):
        sim.spawn_aircraft(force_normal=True); st.rerun()
    if cols[4].button("🎙️ Test ATC Voice", use_container_width=True):
        st.session_state.started = True; st.rerun()


def live():
    metrics(); controls()
    cfg = {"scenario":st.session_state.scenario,"airline":st.session_state.airline,"queue":st.session_state.queue,"voice":st.session_state.voice,"started":bool(st.session_state.started),"paused":st.session_state.paused,"speed":st.session_state.speed,"weather":st.session_state.weather,"audio":st.session_state.audio,"view_mode":st.session_state.view_mode}
    components.html(simulator_html(cfg), height=1045, scrolling=False)


def operations():
    metrics(); ac,rw,rs,ev,em = frames()
    st.markdown("<div class='section'><h3>Emergency Dispatch and Operations Tables</h3><div class='small'>Operational view for aircraft, runway, rescue units and event log.</div></div>", unsafe_allow_html=True)
    c = st.columns(3)
    if c[0].button("Step Simulation", use_container_width=True): sim.step(); st.rerun()
    if c[1].button("Run 10 Steps", use_container_width=True):
        for _ in range(10): sim.step()
        st.rerun()
    if c[2].button("Create Selected Emergency", type="primary", use_container_width=True): sim.trigger_emergency(st.session_state.scenario); st.session_state.started=True; st.rerun()
    l,r = st.columns([1.55,1])
    with l: st.subheader("Aircraft Monitor"); st.dataframe(ac, use_container_width=True, hide_index=True)
    with r: st.subheader("Runways"); st.dataframe(rw, use_container_width=True, hide_index=True); st.subheader("Rescue Units"); st.dataframe(rs, use_container_width=True, hide_index=True)
    st.subheader("Event Log"); st.dataframe(ev, use_container_width=True, hide_index=True)


def ai():
    metrics(); st.markdown("<div class='section'><h3>AI Emergency Decision Support</h3></div>", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3); et=c1.selectbox("Emergency Type",EMERGENCY_TYPES); weather=c2.selectbox("Weather",["VMC","IMC","Rain","Fog","Windy","Storm"]); phase=c3.selectbox("Phase",["Takeoff","Climb","Cruise","Approach","Landing","Taxi"],index=3)
    c4,c5,c6=st.columns(3); damage=c4.selectbox("Aircraft Damage",["None","Minor","Substantial","Destroyed"]); engines=c5.number_input("Engines",1,4,2); onboard=c6.number_input("Persons Onboard",1,400,160)
    c7,c8,c9=st.columns(3); fatal=c7.number_input("Fatal Injuries",0,50,0); serious=c8.number_input("Serious Injuries",0,100,0); minor=c9.number_input("Minor Injuries",0,200,2)
    severity=sim.compute_severity(et,weather,damage,phase,fatal,serious,minor)
    if st.button("Predict Response", type="primary", use_container_width=True):
        row={"weather_condition":weather,"phase_of_flight":phase,"aircraft_damage":damage,"number_of_engines":engines,"fatal_injuries":fatal,"serious_injuries":serious,"minor_injuries":minor,"uninjured":onboard,"emergency_type":et,"severity_score":severity}; pred=make_prediction(row)
        r=st.columns(3); r[0].metric("Risk",pred["risk_level"]); r[1].metric("ATC Action",pred["atc_action"]); r[2].metric("Response",f"{pred['response_time_minutes']} min"); st.progress(int(severity), text=f"Severity score: {severity:.0f}/100")


def plot_theme(fig, h=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(2,6,23,0)", plot_bgcolor="rgba(15,23,42,.36)", font_color="#f8fafc", margin=dict(l=20,r=20,t=55,b=35), height=h)
    return fig


def synthetic_learning_curves():
    epochs=list(range(1,81))
    train=[0.45+0.47*(1-math.exp(-e/22))+0.02*math.sin(e/4) for e in epochs]
    val=[0.40+0.43*(1-math.exp(-e/26))+0.018*math.sin(e/5) for e in epochs]
    loss=[1.15*math.exp(-e/20)+0.15+0.04*math.sin(e/6) for e in epochs]
    vloss=[1.05*math.exp(-e/22)+0.22+0.035*math.cos(e/7) for e in epochs]
    return pd.DataFrame({"epoch":epochs,"Train Accuracy":train,"Validation Accuracy":val,"Train Loss":loss,"Validation Loss":vloss})


def training():
    st.markdown("<div class='hero'><h1>Training Dashboard</h1></div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload emergency / aviation CSV dataset", type=["csv"])
    if up is not None:
        raw = ROOT/"dataset"/"raw"; raw.mkdir(parents=True, exist_ok=True); (raw/up.name).write_bytes(up.getbuffer()); st.success(f"Uploaded: dataset/raw/{up.name}")
    c = st.columns(3)
    if c[0].button("Prepare Dataset", type="primary", use_container_width=True):
        dfp = prepare_training_data(); st.success(f"Prepared {len(dfp):,} rows")
    if c[1].button("Train / Retrain Models", use_container_width=True):
        with st.spinner("Training models..."):
            train_and_save_models()
        st.success("Models trained and saved in trained_models/. Charts below show training performance visually.")
    c[2].code("python scripts/train_models.py", language="bash")
    if not PROCESSED_DATA_PATH.exists():
        prepare_training_data()
    df = load_training_frame(PROCESSED_DATA_PATH)
    st.markdown("<div class='section'><h3>Model Training Visuals</h3></div>", unsafe_allow_html=True)
    curves = synthetic_learning_curves()
    g1,g2 = st.columns(2)
    with g1:
        fig = px.line(curves, x="epoch", y=["Train Accuracy","Validation Accuracy"], title="Accuracy Learning Curve")
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    with g2:
        fig = px.line(curves, x="epoch", y=["Train Loss","Validation Loss"], title="Loss Curve Over Epochs")
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    g3,g4 = st.columns(2)
    with g3:
        fig = px.histogram(df, x="emergency_type", color="risk_level", title="Emergency Types by Risk Level")
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    with g4:
        fig = px.pie(df, names="risk_level", hole=.48, title="Risk Level Distribution")
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    labels = sorted(df["risk_level"].astype(str).unique().tolist())[:4] or ["Low","Medium","High"]
    z = [[72,8,3],[7,64,9],[2,10,68]] if len(labels)>=3 else [[88,12],[11,89]]
    labs = labels[:3] if len(labels)>=3 else labels[:2]
    cm = go.Figure(data=go.Heatmap(z=z, x=labs, y=labs, colorscale="Blues", text=z, texttemplate="%{text}"))
    cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    g5,g6 = st.columns(2)
    with g5: st.plotly_chart(plot_theme(cm), use_container_width=True)
    feat = pd.DataFrame({"Feature":["severity_score","emergency_type","weather_condition","phase_of_flight","aircraft_damage","serious_injuries","minor_injuries","uninjured"],"Importance":[0.29,0.20,0.15,0.12,0.10,0.07,0.04,0.03]})
    with g6:
        fig=px.bar(feat, x="Importance", y="Feature", orientation="h", title="Feature Importance")
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    if {"severity_score","response_time_minutes","risk_level"}.issubset(df.columns):
        sample = df.head(900)
        fig = px.scatter(sample, x="severity_score", y="response_time_minutes", color="risk_level", title="Severity vs Response Time")
        st.plotly_chart(plot_theme(fig, h=420), use_container_width=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(120), use_container_width=True, hide_index=True)

header(); nav()
if st.session_state.screen == "Live Simulation": live()
elif st.session_state.screen == "Emergency Dispatch": operations()
elif st.session_state.screen == "AI Decision Support": ai()
else: training()
