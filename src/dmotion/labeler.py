"""A private, local browser UI for reviewing training pictures."""

import json
import mimetypes
import secrets
import threading
import webbrowser
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .dataset import Dataset

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dMotion — review training pictures</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#11141a;color:#eef0f5}
*{box-sizing:border-box}body{max-width:1120px;margin:auto;padding:24px}
h1{margin:0 0 8px;font-size:25px}p{line-height:1.5;margin:8px 0;color:#bac2d3}
.top,.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.top{justify-content:space-between}
button{font:inherit;border:1px solid #536079;border-radius:8px;padding:11px 16px;
background:#263045;color:inherit;cursor:pointer}button:hover{background:#364763}
button:disabled{opacity:.4;cursor:default}.primary{background:#244da4;border-color:#7396eb}
.primary:hover{background:#3765c7}.workspace{margin:18px 0;background:#080a0e;
border:1px solid #394257;border-radius:10px;padding:10px}
canvas{width:auto;height:auto;max-width:100%;max-height:55vh;margin:auto;
display:block;cursor:crosshair;touch-action:none}
.meta{font-size:13px;color:#9aa8c0;overflow-wrap:anywhere;margin:10px 0}
#notice{min-height:24px;color:#addcb2}#notice.error{color:#ffb0b0}
#counter{font-variant-numeric:tabular-nums}#status{font-weight:600;color:#dde6ff}
.instructions{border-left:3px solid #7396eb;padding-left:13px;margin-top:16px}
.shortcut{font-size:13px}.empty{padding:40px;text-align:center}kbd{color:#dae5ff}
.suggestion{border:1px solid #a58c4e;background:#30281b;color:#ffe6a9;
border-radius:8px;padding:10px 13px;margin:12px 0}
@media(max-width:600px){body{padding:12px}button{padding:10px}.row{gap:6px}}
</style></head><body>
<div class="top"><div><h1>Review training pictures</h1><p id="progress">Loading…</p></div>
<button id="finish">Finish labeling</button></div>
<p class="instructions"><strong>Draw a tight box around each cash bundle or single bill.</strong>
Include all visible notes in a fan or stack. For separate bundles, draw separate boxes.
Money fans, stacks and single bills being displayed all count as <strong>Cash</strong>.</p>
<p>Cards, receipts, phones and empty hands count as <strong>No cash</strong>.
Skip pictures that are too blurry, unclear or unsuitable.</p>
<div class="top"><span id="counter"></span><span id="status"></span></div>
<div id="suggestion" class="suggestion" role="status" aria-live="polite" hidden></div>
<div class="workspace"><canvas id="canvas" width="1000" height="563" tabindex="0"></canvas>
<div id="empty" class="empty" hidden>No pictures to review.</div></div>
<div class="row"><button id="positive" class="primary">Save cash + next</button>
<button id="negative">No cash + next</button><button id="excluded">Skip + next</button>
<button id="undo">Undo box</button><button id="clear">Clear boxes</button></div>
<div class="meta" id="metadata"></div><div id="notice" role="status" aria-live="polite"></div>
<div class="row"><button id="previous">← Previous</button><button id="next">Next →</button>
<button id="unreviewed">Next unreviewed</button></div>
<p class="shortcut">Shortcuts: <kbd>Enter</kbd> save cash · <kbd>0</kbd> no cash ·
<kbd>← →</kbd> browse · <kbd>Delete</kbd> clear boxes.
Everything is saved locally. You can return and change any label.</p>
<script nonce="__TOKEN__">
'use strict';
const token=__TOKEN_JSON__,canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const get=id=>document.getElementById(id),buttons=[...document.querySelectorAll('button')];
let records=[],index=0,boxes=[],photo=null,start=null,draft=null,busy=true,loadVersion=0;
function notice(text,error=false){
 get('notice').textContent=text;get('notice').className=error?'error':'';
}
function controls(disabled){busy=disabled;buttons.forEach(button=>button.disabled=disabled);}
function progress(){
 const reviewed=records.filter(r=>r.status==='positive'||r.status==='negative').length;
 const positive=records.filter(r=>r.status==='positive').length;
 const negative=records.filter(r=>r.status==='negative').length;
 const pending=records.filter(r=>r.status==='unreviewed').length;
 get('progress').textContent=`${reviewed} reviewed · ${positive} with cash · `+
  `${negative} no cash · ${pending} still to review`;
}
function render(){
 ctx.clearRect(0,0,canvas.width,canvas.height);
 if(!photo)return;
 ctx.drawImage(photo,0,0,canvas.width,canvas.height);
 ctx.strokeStyle='#ffffff';ctx.lineWidth=Math.max(2,canvas.width/350);
 const all=draft?[...boxes,draft]:boxes;
 all.forEach((box,i)=>{
   const [x1,y1,x2,y2]=box;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
   ctx.font=`${Math.max(16,canvas.width/55)}px system-ui`;
   ctx.fillStyle='#152340';ctx.fillRect(x1,y1,100,25);
   ctx.fillStyle='#ffffff';ctx.fillText(`Cash ${i+1}`,x1+6,y1+18);
 });
}
function point(event){
 const rect=canvas.getBoundingClientRect();
 return [Math.max(0,Math.min(canvas.width,(event.clientX-rect.left)*canvas.width/rect.width)),
 Math.max(0,Math.min(canvas.height,(event.clientY-rect.top)*canvas.height/rect.height))];
}
function rectangle(a,b){return [Math.round(Math.min(a[0],b[0])),Math.round(Math.min(a[1],b[1])),
 Math.round(Math.max(a[0],b[0])),Math.round(Math.max(a[1],b[1]))];}
canvas.addEventListener('pointerdown',event=>{
 if(busy||!photo||event.button!==0)return;
 canvas.focus({preventScroll:true});
 start=point(event);draft=null;canvas.setPointerCapture(event.pointerId);event.preventDefault();
});
canvas.addEventListener('pointermove',event=>{if(start){draft=rectangle(start,point(event));render();}});
canvas.addEventListener('pointerup',event=>{
 if(!start)return;const box=rectangle(start,point(event));start=null;draft=null;
  if(box[2]-box[0]>2&&box[3]-box[1]>2){
   boxes.push(box);notice('Box drawn. Save cash to keep this label.');
  }
 render();
});
canvas.addEventListener('pointercancel',()=>{start=null;draft=null;render();});
async function api(path,options={}){
 const response=await fetch(path,{...options,cache:'no-store',headers:{'X-Review-Token':token,
 ...(options.body?{'Content-Type':'application/json'}:{}),...options.headers}});
 const data=await response.json();
 if(!response.ok)throw new Error(data.error||'Could not save');return data;
}
async function show(position){
 if(!records.length){photo=null;canvas.hidden=true;get('empty').hidden=false;controls(true);
 get('finish').disabled=false;progress();return;}
 index=Math.max(0,Math.min(records.length-1,position));const record=records[index];
 const version=++loadVersion;controls(true);photo=null;
 boxes=record.review_boxes.map(box=>[...box]);
 start=null;draft=null;render();notice('Loading picture…');
 get('counter').textContent=`Picture ${index+1} / ${records.length}`;
 get('status').textContent={positive:'Saved: cash',negative:'Saved: no cash',excluded:'Skipped',
 unreviewed:'Needs review'}[record.status];
 const suggested=record.status==='unreviewed'&&record.suggestion;
 get('suggestion').hidden=!suggested;
 if(suggested){
  get('status').textContent='Needs review · AI suggestions';
  get('suggestion').textContent=boxes.length?
   `AI proposed ${boxes.length} cash box${boxes.length===1?'':'es'}. Check for missed cash and `+
    'adjust any incorrect boxes, then click Save cash to accept. These are not saved labels yet.':
   'AI found no cash. Check the picture before choosing No cash, or draw boxes around any '+
    'cash it missed. This picture still needs your review.';
 }
 get('metadata').textContent=`Source: ${record.source||'Not supplied'} · Session: ${record.group}`;
 const image=new Image();
 image.onload=()=>{if(version!==loadVersion)return;photo=image;canvas.width=record.width;
 canvas.height=record.height;render();controls(false);notice('');progress();};
 image.onerror=()=>{if(version!==loadVersion)return;controls(false);
  get('positive').disabled=true;get('negative').disabled=true;
  notice('Could not load this picture. Skip it or choose another.',true);};
 image.src=`/api/image/${encodeURIComponent(record.id)}?token=${encodeURIComponent(token)}`;
}
async function save(status){
 if(busy)return;
  if(status==='positive'&&!boxes.length){
   notice('First draw a tight box around each cash bundle or single bill.',true);return;
  }
 controls(true);
 try{
  const saved=await api('/api/review',{method:'POST',body:JSON.stringify({id:records[index].id,
   status,boxes:status==='positive'?boxes:[]})});records[index]=saved;
  progress();
  if(index<records.length-1){await show(index+1);}else{await show(index);
    notice('Saved the last picture. Use Next unreviewed to check any remaining pictures.');}
 }catch(error){controls(false);notice(error.message,true);}
}
get('positive').onclick=()=>save('positive');get('negative').onclick=()=>save('negative');
get('excluded').onclick=()=>save('excluded');
get('undo').onclick=()=>{
 boxes.pop();render();notice('Changes are kept when you click Save cash.');
};
get('clear').onclick=()=>{
 boxes=[];render();notice('Boxes cleared. Draw again, or choose No cash.');
};
get('previous').onclick=()=>show(index-1);get('next').onclick=()=>show(index+1);
get('unreviewed').onclick=()=>{
 for(let offset=1;offset<=records.length;offset++){
  const position=(index+offset)%records.length;
  if(records[position].status==='unreviewed'){show(position);return;}}
 notice('Every picture has been reviewed or skipped. Your labels are ready to build a dataset.');
};
get('finish').onclick=async()=>{
 controls(true);try{await api('/api/stop',{method:'POST',body:'{}'});
 notice('Labels saved. You can close this tab and return to dMotion.');}
 catch(error){controls(false);notice(error.message,true);}
};
document.addEventListener('keydown',event=>{
 if(busy||(event.key==='Enter'&&event.target.tagName==='BUTTON'))return;
 const actions={Enter:()=>save('positive'),'0':()=>save('negative'),
 ArrowLeft:()=>show(index-1),ArrowRight:()=>show(index+1),Delete:()=>get('clear').click()};
 if(actions[event.key]){event.preventDefault();actions[event.key]();}
});
(async()=>{try{records=await api('/api/items');progress();
 const pending=records.findIndex(record=>record.status==='unreviewed');
 await show(pending<0?0:pending);}
 catch(error){notice(error.message,true);get('finish').disabled=false;}})();
</script></body></html>"""


def _review_item(record: dict) -> dict:
    """Offer editable AI boxes without treating a suggestion as a reviewed label."""
    item = deepcopy(record)
    suggestion = item.get("suggestion")
    if item["status"] == "unreviewed" and suggestion is not None:
        item["review_boxes"] = deepcopy(suggestion["boxes"])
    else:
        item["review_boxes"] = deepcopy(item["boxes"])
    return item


def _handler(dataset: Dataset, token: str) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def _respond(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'self'; script-src 'nonce-{token}'; style-src 'unsafe-inline'; "
                "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status: int, value: object) -> None:
            self._respond(status, json.dumps(value).encode("utf-8"), "application/json")

        def _trusted(self, *, image: bool = False) -> bool:
            expected_host = f"127.0.0.1:{self.server.server_address[1]}"
            if self.headers.get("Host") != expected_host:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid local host"})
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{expected_host}":
                self._json(HTTPStatus.FORBIDDEN, {"error": "Requests must come from the labeler"})
                return False
            supplied = self.headers.get("X-Review-Token", "")
            if image:
                supplied = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not secrets.compare_digest(supplied, token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "Missing review token"})
                return False
            return True

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                expected_host = f"127.0.0.1:{self.server.server_address[1]}"
                if self.headers.get("Host") != expected_host:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid local host"})
                    return
                page = PAGE.replace("__TOKEN_JSON__", json.dumps(token)).replace("__TOKEN__", token)
                self._respond(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/items":
                if self._trusted():
                    with lock:
                        records = [_review_item(record) for record in dataset.records()]
                    self._json(HTTPStatus.OK, records)
            elif path.startswith("/api/image/"):
                if not self._trusted(image=True):
                    return
                try:
                    with lock:
                        record = dataset.get_record(unquote(path.removeprefix("/api/image/")))
                        image_path = dataset.image_path(record)
                    content_type = mimetypes.guess_type(image_path.name)[0]
                    if content_type is None or not content_type.startswith("image/"):
                        raise ValueError("Not an image file")
                    self._respond(HTTPStatus.OK, image_path.read_bytes(), content_type)
                except (KeyError, ValueError, FileNotFoundError):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Picture not found"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._trusted():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected JSON")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                path = urlsplit(self.path).path
                if path == "/api/review":
                    if set(value) != {"id", "status", "boxes"}:
                        raise ValueError("Review requires id, status and boxes")
                    if not isinstance(value["id"], str):
                        raise ValueError("Invalid picture ID")
                    with lock:
                        record = dataset.review(
                            value["id"], status=value["status"], boxes=value["boxes"]
                        )
                    self._json(HTTPStatus.OK, _review_item(record))
                elif path == "/api/stop":
                    self._json(HTTPStatus.OK, {"saved": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ReviewHandler


def run_labeler(directory: Path, *, open_browser: bool = True, port: int = 0) -> int:
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    dataset = Dataset(directory)
    if not dataset.records():
        raise ValueError("No training pictures yet. Record, import or download some images first.")
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(dataset, token))
    address = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Labeler: {address}")
    print("Check AI suggestions or draw cash boxes. Finish labeling to close the local server.")
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nLabels saved. Labeler closed.")
    finally:
        server.server_close()
    return 0
